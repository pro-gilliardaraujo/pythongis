"""
Módulo de Análise de Espaçamento de Plantio
===========================================

Este script processa dados geoespaciais (Shapefiles) de operações agrícolas para:
1. Identificar passadas de máquinas (linhas de plantio/aplicação).
2. Calcular o espaçamento lateral entre essas passadas.
3. Gerar estatísticas de qualidade operacional (espaçamento médio, falhas).
4. Produzir mapas interativos para visualização dos resultados.

Refatoração seguindo princípios de Código Limpo e Design Centrado no Humano.
"""

import os
import math
import glob
import zipfile
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
from shapely.ops import substring, unary_union, nearest_points
import folium
from folium.plugins import MeasureControl, Fullscreen

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

class ConfiguracaoAnalise:
    """Centraliza todos os parâmetros e constantes da análise."""
    
    def __init__(self):
        # Visualização
        self.MAPA_ZOOM_INICIAL = 18
        self.MAPA_MAX_ZOOM = 24
        self.MAPA_LIMITAR_VISAO = True
        
        # Camadas do Mapa
        self.GERAR_CAMADAS_OCULTAS = False
        self.LAYER_NAMES = {
            'pontos': 'Pontos (Original)',
            'linhas': 'Linhas (Calculadas)',
            'cotas_ok': 'Cotas (OK)',
            'cotas_alerta': 'Cotas (Fora)',
            'heatmap': 'Mapa de Tendência'
        }
        self.LAYER_VISIBILITY = {
            'pontos': False,
            'linhas': False,
            'cotas_ok': False,
            'cotas_alerta': True,
            'heatmap': True
        }
        
        # Estilização
        self.ESTILO = {
            'ponto_raio': 1,
            'cor_ponto_padrao': '#FF4500',
            'cor_ponto_ativo': '#00FF00',
            'cor_ponto_inativo': '#FF0000',
            'cor_linha': '#00FFFF',
            'linha_espessura': 3,
            'linha_opacidade': 0.6,
            'cor_cota_ok': '#00FF00',
            'cor_cota_alerta': '#FF0000',
            'heatmap_espessura': 5,
            'heatmap_opacidade': 0.7
        }
        
        # Tolerâncias de Qualidade
        self.ALVO_ESPACAMENTO = 3.0 # Implícito pela média, mas usado para ref
        self.TOLERANCIA_MIN = 2.9
        self.TOLERANCIA_MAX = 3.1
        self.FATOR_CORTE_VISUALIZACAO = 1.5 # Gaps maiores que 1.5x o máx são ignorados
        
        # Algoritmo de Detecção de Passadas (Trajetória)
        self.QUEBRA_TEMPO_SEC = 60
        self.QUEBRA_ANGULO_GRAUS = 60
        self.MIN_MOVIMENTO_DIRECAO = 2.0
        
        # Algoritmo de Espaçamento
        self.OFFSET_CABECEIRA = 0.0
        self.AMOSTRAGEM_METROS = 5.0
        self.MIN_DIST_COTAS_MESMA_LINHA = 10.0
        self.FILTRO_DIST_MIN = 2.0
        self.FILTRO_DIST_MAX = 50.0
        self.MIN_DIST_TOPOLOGICA = 50.0
        
        # Clustering e União
        self.DISTANCIA_CLUSTER = 50.0
        self.MIN_METROS_CLUSTER = 1000.0
        self.MIN_TIRO_STATS = 50.0
        self.MAX_DIST_EMENDA = 50.0
        self.MAX_DIST_EMENDA_LONGA = 1600.0
        self.MAX_ANGULO_EMENDA = 45.0
        
        # Output
        self.PASSOS_VISUALIZACAO = [10] # 1 a cada X cotas

TABELA_CONFIG = ConfiguracaoAnalise()

# ==============================================================================
# UTILITÁRIOS DE GEOMETRIA E ARQUIVOS
# ==============================================================================

class UtilitariosGeo:
    @staticmethod
    def calcular_epsg_utm(lat: float, lon: float) -> int:
        """Determina o código EPSG da zona UTM para uma coordenada Lat/Lon."""
        zone = math.floor((lon + 180) / 6) + 1
        base = 32600 if lat >= 0 else 32700
        return base + zone

    @staticmethod
    def calcular_angulo_entre_vetores(v1: tuple, v2: tuple) -> float:
        """Calcula o ângulo em graus entre dois vetores (dx, dy)."""
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 * mag2 == 0: return 0
        cos_theta = dot / (mag1 * mag2)
        # Clamp para evitar erros de ponto flutuante fora de [-1, 1]
        cos_theta = max(min(cos_theta, 1), -1) 
        return math.degrees(math.acos(cos_theta))

class GerenciadorArquivos:
    """Responsável por encontrar, extrair e carregar arquivos."""
    
    @staticmethod
    def buscar_zip_mais_recente(diretorio: str) -> Optional[str]:
        arquivos = glob.glob(os.path.join(diretorio, "*.zip"))
        if not arquivos:
            return None
        # Ordena por data de modificação (mais recente primeiro)
        arquivos.sort(key=os.path.getmtime, reverse=True)
        return arquivos[0]

    @staticmethod
    def extrair_zip(caminho_zip: str, pasta_destino: str) -> str:
        os.makedirs(pasta_destino, exist_ok=True)
        with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
            zip_ref.extractall(pasta_destino)
        return pasta_destino

    @staticmethod
    def encontrar_shapefile(diretorio: str) -> Optional[str]:
        arquivos = glob.glob(os.path.join(diretorio, "**", "*.shp"), recursive=True)
        return arquivos[0] if arquivos else None

# ==============================================================================
# LÓGICA DE PROCESSAMENTO DE DADOS
# ==============================================================================

class ProcessadorTrajetoria:
    """
    Transforma nuvens de pontos GPS em linhas de trajetória contínuas (passadas).
    """
    
    @staticmethod
    def gerar_linhas_de_voo(gdf_pontos: gpd.GeoDataFrame, config: ConfiguracaoAnalise) -> Tuple[List[LineString], List[dict]]:
        """
        Segmenta pontos em linhas baseando-se em quebras de tempo e direção.
        """
        coluna_tempo = 'Time'
        if coluna_tempo not in gdf_pontos.columns:
            return [], []

        pontos = gdf_pontos.copy()
        # Garante ordenação temporal
        pontos[coluna_tempo] = pd.to_datetime(pontos[coluna_tempo], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        pontos = pontos.dropna(subset=[coluna_tempo]).sort_values(coluna_tempo).reset_index(drop=True)
        
        if len(pontos) < 2:
            return [], []

        segmentos_indices = []
        segmento_atual = [pontos.index[0]]
        angulo_anterior = None
        
        # Percorre pontos calculando deltas
        for i in range(1, len(pontos)):
            pt_atual = pontos.geometry.iloc[i]
            pt_ant = pontos.geometry.iloc[i-1]
            tempo_atual = pontos[coluna_tempo].iloc[i]
            tempo_ant = pontos[coluna_tempo].iloc[i-1]
            
            # Delta Tempo
            delta_sec = (tempo_atual - tempo_ant).total_seconds()
            
            # Delta Espaço/Direção
            dx = pt_atual.x - pt_ant.x
            dy = pt_atual.y - pt_ant.y
            distancia = math.hypot(dx, dy)
            angulo = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            
            deve_quebrar = False
            
            # 1. Quebra por Tempo (gap grande, ex: manobra longa ou parada)
            if delta_sec > config.QUEBRA_TEMPO_SEC:
                deve_quebrar = True
            # 2. Quebra por Ângulo (curva fechada, ex: retomada de linha)
            elif distancia >= config.MIN_MOVIMENTO_DIRECAO and angulo_anterior is not None:
                diff_angle = abs(angulo - angulo_anterior)
                diff_angle = min(diff_angle, 360 - diff_angle)
                if diff_angle > config.QUEBRA_ANGULO_GRAUS:
                    deve_quebrar = True
            
            if deve_quebrar:
                if len(segmento_atual) >= 2:
                    segmentos_indices.append(segmento_atual)
                segmento_atual = [pontos.index[i]]
                angulo_anterior = angulo if distancia >= config.MIN_MOVIMENTO_DIRECAO else None
            else:
                segmento_atual.append(pontos.index[i])
                if distancia >= config.MIN_MOVIMENTO_DIRECAO:
                    angulo_anterior = angulo
                    
        # Adiciona o último
        if len(segmento_atual) >= 2:
            segmentos_indices.append(segmento_atual)
            
        # Constrói Geometrias LineString
        linhas_geom = []
        metadados = []
        
        for indices in segmentos_indices:
            try:
                # Pega os pontos correspondentes aos índices
                pts_segmento = pontos.loc[indices].geometry.tolist()
                linha = LineString(pts_segmento)
                timestamp_inicio = pontos.loc[indices[0], coluna_tempo]
                
                linhas_geom.append(linha)
                metadados.append({'time': timestamp_inicio})
            except Exception:
                continue
                
        return linhas_geom, metadados

    @staticmethod
    def agrupar_por_clusters(gdf_linhas: gpd.GeoDataFrame, config: ConfiguracaoAnalise) -> gpd.GeoDataFrame:
        """
        Agrupa linhas próximas para identificar 'talhões' ou áreas de trabalho distintas.
        Remove ruídos (linhas isoladas muito curtas).
        """
        if gdf_linhas.empty:
            return gdf_linhas

        # Cria buffer para unir linhas próximas
        buffers = gdf_linhas.geometry.buffer(config.DISTANCIA_CLUSTER)
        uniao = unary_union(buffers)
        
        poligonos_clusters = []
        if uniao.geom_type == 'MultiPolygon':
            poligonos_clusters = list(uniao.geoms)
        else:
            poligonos_clusters = [uniao]
            
        gdf_clusters = gpd.GeoDataFrame(geometry=poligonos_clusters, crs=gdf_linhas.crs)
        gdf_clusters['cluster_id'] = range(len(gdf_clusters))
        
        # Associa cada linha ao seu cluster
        gdf_linhas_cluster = gpd.sjoin(gdf_linhas, gdf_clusters, how='left', predicate='intersects')
        
        # Filtra clusters irrelevantes (comprimento total de linhas muito baixo)
        gdf_linhas_cluster['comp_temp'] = gdf_linhas_cluster.geometry.length
        stats = gdf_linhas_cluster.groupby('cluster_id')['comp_temp'].sum().reset_index()
        
        ids_validos = stats[stats['comp_temp'] >= config.MIN_METROS_CLUSTER]['cluster_id']
        
        # Filtra e organiza
        gdf_final = gdf_linhas_cluster[gdf_linhas_cluster['cluster_id'].isin(ids_validos)].copy()
        
        # Gera labels legíveis (A, B, C...) ordenados por tempo
        if not gdf_final.empty:
            meta_tempo = gdf_final.groupby('cluster_id')['time'].min().sort_values()
            mapa_labels = {}
            for i, cluster_id in enumerate(meta_tempo.index):
                # Gera A, B, ... Z, AA...
                letra = chr(65 + i) if i < 26 else f"Z{i}"
                mapa_labels[cluster_id] = letra
                
            gdf_final['cluster_label'] = gdf_final['cluster_id'].map(mapa_labels)
            
        return gdf_final.drop(columns=['comp_temp', 'index_right'])

    @staticmethod
    def conectar_segmentos_quebrados(gdf_linhas: gpd.GeoDataFrame, config: ConfiguracaoAnalise) -> gpd.GeoDataFrame:
        """
        Une segmentos sequenciais que foram quebrados artificialmente (ex: falha GPS momentânea).
        Analisa ponta-a-ponta e alinhamento vetorial.
        """
        if gdf_linhas.empty or 'cluster_label' not in gdf_linhas.columns:
            return gdf_linhas
            
        linhas_unificadas = []
        
        for label in gdf_linhas['cluster_label'].unique():
            if pd.isna(label): continue
            
            # Processa cada talhão separadamente, ordenado por tempo
            grupo = gdf_linhas[gdf_linhas['cluster_label'] == label].sort_values('time')
            lista_linhas = list(grupo.itertuples(index=False))
            
            if not lista_linhas: continue
            
            # Inicia com a primeira linha
            geom_atual = lista_linhas[0].geometry
            meta_atual = lista_linhas[0]._asdict()
            
            for i in range(1, len(lista_linhas)):
                prox_geom = lista_linhas[i].geometry
                prox_meta = lista_linhas[i]._asdict()
                
                # Coordenadas das pontas
                # Precisa ter pelo menos 2 pontos para ter vetor
                if len(geom_atual.coords) < 2 or len(prox_geom.coords) < 2:
                    # Fallback distância simples
                    if Point(geom_atual.coords[-1]).distance(Point(prox_geom.coords[0])) <= config.MAX_DIST_EMENDA:
                         geom_atual = LineString(list(geom_atual.coords) + list(prox_geom.coords))
                    else:
                         # Salva atual e inicia nova
                         meta_salvar = meta_atual.copy(); meta_salvar['geometry'] = geom_atual
                         linhas_unificadas.append(meta_salvar)
                         geom_atual = prox_geom; meta_atual = prox_meta
                    continue

                # Análise Vetorial
                p_fim_atual = geom_atual.coords[-1]
                p_inicio_prox = prox_geom.coords[0]
                
                # Vetor Gap (Salto)
                dx_gap = p_inicio_prox[0] - p_fim_atual[0]
                dy_gap = p_inicio_prox[1] - p_fim_atual[1]
                dist_gap = math.hypot(dx_gap, dy_gap)
                
                # Vetor Linha Atual (Direção Geral)
                p_inicio_atual = geom_atual.coords[0]
                dx_linha = p_fim_atual[0] - p_inicio_atual[0]
                dy_linha = p_fim_atual[1] - p_inicio_atual[1]
                
                angulo_linha = math.degrees(math.atan2(dy_linha, dx_linha))
                angulo_gap = math.degrees(math.atan2(dy_gap, dx_gap))
                
                diff_angular = abs(angulo_linha - angulo_gap)
                diff_angular = min(diff_angular, 360 - diff_angular)
                
                # Decide limite de distância baseado no alinhamento
                # Se estiver muito bem alinhado (<= 15 graus), permite gap maior
                dist_limite = config.MAX_DIST_EMENDA_LONGA if diff_angular <= 15 else config.MAX_DIST_EMENDA
                
                pode_unir = (dist_gap <= dist_limite) and (diff_angular <= config.MAX_ANGULO_EMENDA)
                
                if pode_unir:
                    # Fundir geometrias
                    novas_coords = list(geom_atual.coords) + list(prox_geom.coords)
                    geom_atual = LineString(novas_coords)
                else:
                    # Finaliza segmento e começa novo
                    meta_salvar = meta_atual.copy(); meta_salvar['geometry'] = geom_atual
                    linhas_unificadas.append(meta_salvar)
                    geom_atual = prox_geom; meta_atual = prox_meta
            
            # Salva o remanescente
            meta_salvar = meta_atual.copy(); meta_salvar['geometry'] = geom_atual
            linhas_unificadas.append(meta_salvar)
            
        return gpd.GeoDataFrame(linhas_unificadas, crs=gdf_linhas.crs)

class AnalisadorEspacamento:
    """Núcleo da análise de qualidade (Espaçamento entre passadas)."""
    
    def __init__(self, config: ConfiguracaoAnalise):
        self.config = config
        
    def processar(self, gdf_linhas: gpd.GeoDataFrame) -> dict:
        """
        Executa a varredura de espaçamento:
        1. Amostra pontos ao longo de cada linha.
        2. Busca linha vizinha mais próxima (lateral).
        3. Valida se é um paralelismo real (ângulo ~90º).
        4. Coleta métricas.
        """
        resultados = {
            'cotas_geometria': [], # Linhas representando a medição (visual)
            'cotas_dados': [],     # Valores numéricos
            'heatmap_geometria': [], # Segmentos da linha pintados (qualidade)
            'heatmap_valores': [],
            'todas_distancias': [] # Lista pura para estatística
        }
        
        sindex = gdf_linhas.sindex
        
        for idx, row in gdf_linhas.iterrows():
            linha_geom = row.geometry
            comprimento = linha_geom.length
            
            # Gera pontos de amostragem (ex: a cada 5 metros)
            passo = self.config.AMOSTRAGEM_METROS
            distancias_amostragem = np.arange(0, comprimento, passo)
            if len(distancias_amostragem) == 0: distancias_amostragem = [comprimento/2]
            
            pontos_amostra = [linha_geom.interpolate(d) for d in distancias_amostragem]
            
            ultimo_pos_valida = -9999
            
            for i, ponto_origem in enumerate(pontos_amostra):
                dist_na_linha = distancias_amostragem[i]
                
                # Ignora cabeceiras (início/fim da linha)
                offset = self.config.OFFSET_CABECEIRA
                if dist_na_linha < offset or dist_na_linha > (comprimento - offset):
                    continue
                
                # 1. Busca vizinhos espaciais (Candidatos)
                bbox_busca = ponto_origem.buffer(self.config.FILTRO_DIST_MAX).bounds
                indices_candidatos = list(sindex.intersection(bbox_busca))
                
                melhor_cota = None
                menor_dist = float('inf')
                
                # Vetor Tangente da linha na posição atual (para verificar perpendicularidade)
                # Pega um ponto logo a frente/trás
                delta_t = 1.0
                p_frente = linha_geom.interpolate(dist_na_linha + delta_t) if (dist_na_linha + delta_t) <= comprimento else linha_geom.interpolate(dist_na_linha - delta_t)
                # Se estiver no fim, inverte o vetor para manter lógica (deltas)
                vec_tangente = (p_frente.x - ponto_origem.x, p_frente.y - ponto_origem.y)
                if dist_na_linha + delta_t > comprimento: # Inverteu?
                     vec_tangente = (ponto_origem.x - p_frente.x, ponto_origem.y - p_frente.y)

                for idx_cand in indices_candidatos:
                    candidato = gdf_linhas.iloc[idx_cand]
                    
                    # Encontra o ponto mais próximo na geometria vizinha
                    p_origem_proj, p_vizinho_proj = nearest_points(ponto_origem, candidato.geometry)
                    dist = p_origem_proj.distance(p_vizinho_proj)
                    
                    # Filtros iniciais de distância
                    if dist < self.config.FILTRO_DIST_MIN or dist > self.config.FILTRO_DIST_MAX:
                        continue
                        
                    # Se for a mesma linha, verifica distância "topológica" (linear)
                    # Evita medir consigo mesmo se for uma curva fechada perto
                    if idx_cand == idx:
                        pos_vizinho = candidato.geometry.project(p_vizinho_proj)
                        if abs(pos_vizinho - dist_na_linha) < self.config.MIN_DIST_TOPOLOGICA:
                            continue

                    if dist < menor_dist:
                        # Validação de Ângulo (Deve ser medição lateral/perpendicular)
                        vec_cota = (p_vizinho_proj.x - ponto_origem.x, p_vizinho_proj.y - ponto_origem.y)
                        angulo_incidencia = UtilitariosGeo.calcular_angulo_entre_vetores(vec_tangente, vec_cota)
                        
                        # Aceitamos ângulos entre 45 e 135 (perto de 90)
                        # Se for 0 ou 180, é medição longitudinal (ruído)
                        if 45 <= angulo_incidencia <= 135:
                            menor_dist = dist
                            melhor_cota = p_vizinho_proj

                # Se achou uma cota válida
                if melhor_cota:
                    # Filtro Final de Qualidade Visual (Descarte de Gaps absurdos)
                    limite_visual = self.config.TOLERANCIA_MAX * self.config.FATOR_CORTE_VISUALIZACAO
                    if menor_dist > limite_visual:
                        continue
                        
                    # Registro para Heatmap (Qualidade do Trecho)
                    # Recorta um pedaço da linha original
                    meio_passo = passo / 2
                    trecho_geom = substring(linha_geom, max(0, dist_na_linha-meio_passo), min(comprimento, dist_na_linha+meio_passo))
                    resultados['heatmap_geometria'].append(trecho_geom)
                    resultados['heatmap_valores'].append(menor_dist)
                    
                    # Registro da Cota (Desenho da régua)
                    # Evita poluição visual: cotas muito perto na mesma linha
                    if (dist_na_linha - ultimo_pos_valida) >= self.config.MIN_DIST_COTAS_MESMA_LINHA:
                        resultados['cotas_geometria'].append(LineString([ponto_origem, melhor_cota]))
                        resultados['cotas_dados'].append({'val': menor_dist, 'pos': dist_na_linha})
                        resultados['todas_distancias'].append(menor_dist)
                        ultimo_pos_valida = dist_na_linha
                        
        return resultados

# ==============================================================================
# VISUALIZAÇÃO E RELATÓRIOS
# ==============================================================================

class GeradorRelatorio:
    @staticmethod
    def exibir_terminal(metricas: dict, gdf_linhas: gpd.GeoDataFrame, config: ConfiguracaoAnalise):
        distancias = np.array(metricas['todas_distancias'])
        # Filtro para estatística (remove outliers extremos)
        validos = distancias[(distancias > 0.5) & (distancias < 50.0)]
        
        if len(validos) == 0:
            print("Não foram encontrados dados suficientes para estatística.")
            return
            
        # Estatística Robusta (IQR)
        q1 = np.percentile(validos, 25)
        q3 = np.percentile(validos, 75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        
        robustos = validos[(validos >= limite_inf) & (validos <= limite_sup)]
        if len(robustos) == 0: robustos = validos
        
        media = np.mean(robustos)
        minimo = np.min(validos)
        
        # Tiro Médio (Comprimento das passadas)
        # Filtra toquinhos (manobras)
        tiros = gdf_linhas[gdf_linhas.geometry.length >= config.MIN_TIRO_STATS]
        tiro_medio = tiros.geometry.length.mean() if not tiros.empty else 0
        
        print("\n--- Resultados da Análise ---")
        print(f"Média Estimada: {media:.2f} m")
        print(f"Mínimo Absoluto: {minimo:.2f} m")
        print(f"Tiro Médio (Linhas de Trabalho): {tiro_medio:.1f} m")
        
        return media, tiro_medio

    @staticmethod
    def gerar_html_tabela(media: float, tiro_medio: float, gdf_linhas: gpd.GeoDataFrame, config: ConfiguracaoAnalise) -> str:
        """Gera o HTML flutuante com as estatísticas para o mapa."""
        
        # Tabela Geral
        html = f"""
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr><td><b>Espaçamento médio</b></td><td>{media:.2f} m</td></tr>
            <tr><td><b>Tiro médio (Geral)</b></td><td>{tiro_medio:.2f} m</td></tr>
        </table>
        """
        
        # Tabela por Cluster (Talhão)
        if 'cluster_label' in gdf_linhas.columns:
            html += "<table border='1' style='border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 5px; border-top: none;'>"
            html += "<tr><th>Área</th><th>Tiro Médio*</th></tr>"
            
            labels = sorted(gdf_linhas['cluster_label'].dropna().unique())
            for lab in labels:
                grupo = gdf_linhas[gdf_linhas['cluster_label'] == lab]
                # Filtra tiros curtos
                tiros = grupo[grupo.geometry.length >= config.MIN_TIRO_STATS]
                if tiros.empty: tiros = grupo
                media_local = tiros.geometry.length.mean()
                html += f"<tr><td style='text-align:center;'><b>{lab}</b></td><td style='text-align:left;'>{media_local:.1f}m</td></tr>"
            html += "</table>"
            
        return html

class VisualizadorMapa:
    def __init__(self, config: ConfiguracaoAnalise):
        self.config = config

    def gerar_mapas(self, gdf_dados: gpd.GeoDataFrame, metricas: dict, gdf_linhas: gpd.GeoDataFrame, stat_html: str, pasta_saida: str, nome_base: str):
        """Orquestra a criação dos arquivos HTML."""
        
        # Preparação dos dados em WGS84 (Lat/Lon) para o Leaflet
        gdf_heatmap = gpd.GeoDataFrame({'valor': metricas['heatmap_valores'], 'geometry': metricas['heatmap_geometria']}, crs=gdf_linhas.crs).to_crs(epsg=4326)
        gdf_cotas = gpd.GeoDataFrame({'valor': metricas['cotas_dados'], 'geometry': metricas['cotas_geometria']}, crs=gdf_linhas.crs).to_crs(epsg=4326)
        gdf_linhas_wgs = gdf_linhas.to_crs(epsg=4326)
        
        # Amostragem de pontos para não travar o browser
        gdf_pontos_total = gdf_dados.to_crs(epsg=4326)
        if len(gdf_pontos_total) > 5000:
            gdf_pontos_total = gdf_pontos_total.sample(5000)
            
        centro = [gdf_pontos_total.geometry.y.mean(), gdf_pontos_total.geometry.x.mean()]
        bounds_total = gdf_pontos_total.total_bounds # [minx, miny, maxx, maxy]
        
        # Loop de densidades (Passos)
        passos = self.config.PASSOS_VISUALIZACAO
        
        for passo in passos:
            print(f"Gerando mapa para visualização...")
            
            m = folium.Map(location=centro, zoom_start=self.config.MAPA_ZOOM_INICIAL, tiles=None, max_zoom=self.config.MAPA_MAX_ZOOM)
            
            # Ajuste de visão
            margem = 0.05
            bounds_leaflet = [[bounds_total[1]*(1-margem), bounds_total[0]*(1-margem)], [bounds_total[3]*(1+margem), bounds_total[2]*(1+margem)]]
            m.fit_bounds(bounds_leaflet)
            
            # Base (Satélite)
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Found', name='Satélite', overlay=False
            ).add_to(m)
            
            # --- Camadas ---
            self._adicionar_camada_heatmap(m, gdf_heatmap)
            self._adicionar_camada_cotas(m, gdf_cotas, passo)
            self._adicionar_camada_linhas(m, gdf_linhas_wgs)
            self._adicionar_camada_pontos(m, gdf_pontos_total)
            
            # --- Labels de Clusters ---
            self._adicionar_labels_cluster(m, gdf_linhas_wgs)

            # --- Controles de UI ---
            self._montar_painel_info(m, stat_html)
            
            # Ferramenta de Medição
            MeasureControl(position='topright', primary_length_unit='meters', active_color='#00FF00').add_to(m)
            Fullscreen().add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)

            # Salvar
            nome_arquivo = f"{nome_base}.html"
            if len(passos) > 1:
                nome_arquivo = f"{nome_base} - densidade_{passo}.html"
                
            caminho_final = os.path.join(pasta_saida, nome_arquivo)
            m.save(caminho_final)
            print(f"Mapa salvo em: {caminho_final}")
            
            # Compatibilidade legada: se for passo 1 e tiver multiplos, salva o default também
            if passo == 1 and len(passos) > 1:
                 m.save(os.path.join(pasta_saida, f"{nome_base}.html"))

    def _adicionar_camada_heatmap(self, m, gdf):
        if not self.config.LAYER_VISIBILITY['heatmap']: return
        fg = folium.FeatureGroup(name=self.config.LAYER_NAMES['heatmap'], show=True)
        
        for _, row in gdf.iterrows():
            val = row['valor']
            cor = self.config.ESTILO['cor_cota_alerta'] if (val < self.config.TOLERANCIA_MIN or val > self.config.TOLERANCIA_MAX) else self.config.ESTILO['cor_cota_ok']
            coords = [(pt[1], pt[0]) for pt in row.geometry.coords]
            folium.PolyLine(coords, color=cor, weight=self.config.ESTILO['heatmap_espessura'], opacity=self.config.ESTILO['heatmap_opacidade']).add_to(fg)
        fg.add_to(m)

    def _adicionar_camada_cotas(self, m, gdf, passo_densidade):
        # Separa cotas OK e Alertas (alertas sempre aparecem)
        fg_ok = folium.FeatureGroup(name=self.config.LAYER_NAMES['cotas_ok'], show=False)
        fg_alerta = folium.FeatureGroup(name=self.config.LAYER_NAMES['cotas_alerta'], show=True)
        
        for i, row in gdf.iterrows():
            val = row['valor']['val']
            is_alerta = val < self.config.TOLERANCIA_MIN or val > self.config.TOLERANCIA_MAX
            
            # Lógica de amostragem visual para não poluir
            if not is_alerta and (i % passo_densidade != 0):
                continue
            
            grupo_alvo = fg_alerta if is_alerta else fg_ok
            cor = self.config.ESTILO['cor_cota_alerta'] if is_alerta else self.config.ESTILO['cor_cota_ok']
            
            # Linha da cota
            coords = [(pt[1], pt[0]) for pt in row.geometry.coords]
            folium.PolyLine(coords, color=cor, weight=1, opacity=0.9).add_to(grupo_alvo)
            
            # Marcador com valor
            centro_lat = (coords[0][0] + coords[1][0])/2
            centro_lon = (coords[0][1] + coords[1][1])/2
            
            # Classes CSS para zoom inteligente
            zoom_class = "cota-alert" if is_alerta else "cota-normal"
            style_box = f"background: {'rgba(255,255,255,0.7)' if not is_alerta else cor}; color: {'black' if not is_alerta else 'white'}; border: 1px solid {cor}; border-radius: 4px; padding: 2px"
            
            folium.map.Marker(
                [centro_lat, centro_lon],
                icon=folium.DivIcon(
                    icon_size=(100,20),
                    icon_anchor=(50,10),
                    class_name=f"cota-marker {zoom_class}",
                    html=f'<div style="{style_box}; text-align: center; font-size: 10px; font-weight: bold;">{val:.2f}m</div>'
                )
            ).add_to(grupo_alvo)
            
        fg_alerta.add_to(m)
        fg_ok.add_to(m)

    def _adicionar_camada_linhas(self, m, gdf):
        if not self.config.LAYER_VISIBILITY['linhas']: return
        fg = folium.FeatureGroup(name=self.config.LAYER_NAMES['linhas'], show=False)
        folium.GeoJson(
            gdf, 
            style_function=lambda x: {
                'color': self.config.ESTILO['cor_linha'], 
                'weight': self.config.ESTILO['linha_espessura'], 
                'opacity': self.config.ESTILO['linha_opacidade']
            }
        ).add_to(fg)
        fg.add_to(m)

    def _adicionar_camada_pontos(self, m, gdf):
        # Apenas se explicitamente solicitado
        if not self.config.LAYER_VISIBILITY['pontos']: return
        
        fg = folium.FeatureGroup(name=self.config.LAYER_NAMES['pontos'], show=False)
        for _, row in gdf.iterrows():
             folium.CircleMarker(
                 [row.geometry.y, row.geometry.x],
                 radius=1, color='orange', fill=True
             ).add_to(fg)
        fg.add_to(m)

    def _adicionar_labels_cluster(self, m, gdf):
        if 'cluster_label' not in gdf.columns: return
        # Calcula centróide de cada cluster
        clusters = gdf.dissolve(by='cluster_label')
        for label, row in clusters.iterrows():
            centro = row.geometry.centroid
            folium.Marker(
                [centro.y, centro.x],
                icon=folium.DivIcon(html=f"<div style='font-size: 24px; color: white; font-weight: bold; text-shadow: 2px 2px 4px black;'>{label}</div>")
            ).add_to(m)

    def _montar_painel_info(self, m, html_stats):
        # CSS Injetado para Painel e Controle de Zoom
        css = """
        <style>
            .info-panel {
                position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%);
                z-index: 9999; background: rgba(255,255,255,0.95); padding: 10px;
                border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                min-width: 250px; font-family: sans-serif;
            }
            .leaflet-control-attribution { display: none !important; }
            /* Zoom Logic */
            .hide-cotas .cota-normal { display: none; }
        </style>
        """
        js = f"""
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                var map = {m.get_name()};
                map.on('zoomend', function() {{
                    var z = map.getZoom();
                    var container = map.getContainer();
                    if (z < 19) container.classList.add('hide-cotas');
                    else container.classList.remove('hide-cotas');
                }});
                // Trigger inicial
                map.fire('zoomend');
                
                // Hack para mover layer control (opcional, mantendo simples por agora)
            }});
        </script>
        """
        html_final = f'{css}{js}<div class="info-panel">{html_stats}</div>'
        m.get_root().html.add_child(folium.Element(html_final))

# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    inicio_proc = datetime.now()
    dir_atual = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Obtenção de Dados
    zip_path = GerenciadorArquivos.buscar_zip_mais_recente(dir_atual)
    if zip_path:
        print(f"Arquivo ZIP encontrado: {zip_path}")
        pasta_dados = GerenciadorArquivos.extrair_zip(zip_path, os.path.join(dir_atual, "dados_extraidos"))
    else:
        print("Nenhum ZIP encontrado. Buscando na pasta 'doc'.")
        pasta_dados = os.path.join(dir_atual, "doc")
        
    shp_path = GerenciadorArquivos.encontrar_shapefile(pasta_dados)
    if not shp_path:
        print("Erro: Nenhum arquivo .shp encontrado.")
        return

    print(f"Carregando arquivo: {shp_path}")
    try:
        size_mb = os.path.getsize(shp_path) / (1024*1024)
        print(f"Tamanho do arquivo: {size_mb:.2f} MB")
        gdf_bruto = gpd.read_file(shp_path)
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
        return

    if gdf_bruto.empty: return

    # 2. Pré-processamento e Projeção
    if not gdf_bruto.crs.is_projected:
         centro = gdf_bruto.geometry.iloc[0] # Simplificação
         epsg = UtilitariosGeo.calcular_epsg_utm(centro.y, centro.x)
         print(f"Convertendo para UTM (EPSG:{epsg})...")
         gdf_utm = gdf_bruto.to_crs(epsg=epsg)
    else:
         gdf_utm = gdf_bruto

    # 3. Processamento Core (Trajetória)
    print("Gerando linhas a partir dos pontos (por tempo e direção)...")
    linhas_geom, metadados_linhas = ProcessadorTrajetoria.gerar_linhas_de_voo(gdf_utm, TABELA_CONFIG)
    
    if not linhas_geom:
        print("Não foi possível gerar linhas de trajetória.")
        return
        
    gdf_linhas = gpd.GeoDataFrame(metadados_linhas, geometry=linhas_geom, crs=gdf_utm.crs)
    print(f"Segmentos gerados: {len(gdf_linhas)}")

    print("Agrupando linhas em Clusters (Áreas de Interesse)...")
    gdf_linhas = ProcessadorTrajetoria.agrupar_por_clusters(gdf_linhas, TABELA_CONFIG)
    print(f"Clusters identificados: {gdf_linhas['cluster_id'].nunique() if 'cluster_id' in gdf_linhas else 0}")

    print("Unindo segmentos fragmentados...")
    gdf_linhas = ProcessadorTrajetoria.conectar_segmentos_quebrados(gdf_linhas, TABELA_CONFIG)

    # 4. Análise de Espaçamento
    print("Calculando distâncias laterais...")
    analisador = AnalisadorEspacamento(TABELA_CONFIG)
    metricas = analisador.processar(gdf_linhas)

    # 5. Relatórios e Exportação
    media, tiro_medio = GeradorRelatorio.exibir_terminal(metricas, gdf_linhas, TABELA_CONFIG)
    html_painel = GeradorRelatorio.gerar_html_tabela(media, tiro_medio, gdf_linhas, TABELA_CONFIG)
    
    viz = VisualizadorMapa(TABELA_CONFIG)
    
    ontem_str = (datetime.now() - timedelta(days=1)).strftime('%d-%m-%Y')
    nome_base = f"Mapa Espaçamento Plantio - {ontem_str}"
    pasta_mapas = os.path.join(dir_atual, "mapas")
    os.makedirs(pasta_mapas, exist_ok=True)
    
    viz.gerar_mapas(gdf_utm, metricas, gdf_linhas, html_painel, pasta_mapas, nome_base)

    tempo_total = datetime.now() - inicio_proc
    print(f"\nTempo total de processamento: {str(tempo_total).split('.')[0]}")

if __name__ == "__main__":
    main()
