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
import sys
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
        self.MAPA_MIN_ZOOM = 16 # Mantido para referência, mas não forçado no mapa base se não desejado
        self.MAPA_MAX_ZOOM = 24
        self.MAPA_LIMITAR_VISAO = True
        
        # Opções Visuais
        self.FUNDO_COTAS = False # Se True, exibe retângulo colorido. Se False, apenas texto colorido.
        
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
            'linha_espessura': 1,
            'linha_opacidade': 0.6,
            'cor_cota_ok': '#00FF00',
            'cor_cota_abaixo': '#FF0000',
            'cor_cota_acima': '#FFFF00', # Amarelo (antes era #FFA500 Laranja)
            'heatmap_espessura': 3,
            'heatmap_opacidade': 0.8
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
        self.ZOOM_MIN_COTAS = 18 # Zoom mínimo para exibir cotas
        self.ZOOM_MAX_CLUSTERS = 18 # Zoom máximo para exibir clusters (acima disso, oculta)

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
    def buscar_todos_zips(diretorio: str) -> List[str]:
        """Retorna lista de todos os arquivos ZIP no diretório."""
        arquivos = glob.glob(os.path.join(diretorio, "*.zip"))
        # Ordena por data de modificação (mais recente primeiro)
        arquivos.sort(key=os.path.getmtime, reverse=True)
        return arquivos

    @staticmethod
    def extrair_zip(caminho_zip: str, pasta_base_extracao: str) -> str:
        """Extrai o ZIP para uma subpasta única baseada no nome do arquivo."""
        nome_arquivo = os.path.splitext(os.path.basename(caminho_zip))[0]
        pasta_destino = os.path.join(pasta_base_extracao, nome_arquivo)
        
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
        
        # Tabela Geral (Ajustado para Row Única e Compacta - Label em frente ao valor)
        html = f"""
        <div style="display: flex; justify-content: center; align-items: center; gap: 15px; font-size: 12px; margin-bottom: 5px;">
            <div>
                <b>Espaçamento médio:</b> {media:.2f} m
            </div>
            <div style="width: 1px; height: 12px; background: #ccc;"></div>
            <div>
                <b>Tiro médio (Geral):</b> {tiro_medio:.2f} m
            </div>
        </div>
        """
        
        # Tabela por Cluster (Talhão)
        if 'cluster_label' in gdf_linhas.columns:
            html += "<table style='border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 5px; border-top: none; border: none;'>"
            html += "<tr><th style='text-align:center;'>Área</th><th style='text-align:center;'>Tiro Médio*</th></tr>"
            
            labels = sorted(gdf_linhas['cluster_label'].dropna().unique())
            for lab in labels:
                # Remove prefixo do arquivo para exibir apenas a letra/numero da área
                lab_curto = lab.split(' ')[-1] if ' ' in lab else lab
                
                grupo = gdf_linhas[gdf_linhas['cluster_label'] == lab]
                # Filtra tiros curtos
                tiros = grupo[grupo.geometry.length >= config.MIN_TIRO_STATS]
                if tiros.empty: tiros = grupo
                media_local = tiros.geometry.length.mean()
                html += f"<tr><td style='text-align:center;'><b>{lab}</b></td><td style='text-align:center;'>{media_local:.1f}m</td></tr>"
            html += "</table>"
            
        return html

class VisualizadorMapa:
    def __init__(self, config: ConfiguracaoAnalise):
        self.config = config
        self.mapa = None
        self.html_stats_agregado = ""
        self.bounds_geral = []

    def iniciar_mapa(self, centro_inicial=[0,0], prefer_canvas=False):
        """Cria a instância base do mapa."""
        self.mapa = folium.Map(
            location=centro_inicial,
            zoom_start=self.config.MAPA_ZOOM_INICIAL,
            tiles=None,
            max_zoom=self.config.MAPA_MAX_ZOOM,
            zoom_control=False,
            prefer_canvas=prefer_canvas
        )
        
        # Meta Tag para Mobile (Crítico para evitar fundo branco/escala errada)
        meta = """
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
        """
        self.mapa.get_root().header.add_child(folium.Element(meta))
        
        # Adiciona TileLayer IMEDIATAMENTE (Garante que o fundo carregue independente dos overlays)
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google', name='Satélite', overlay=False, control=False,
            max_native_zoom=20,
            max_zoom=self.config.MAPA_MAX_ZOOM
        ).add_to(self.mapa)
        
    def adicionar_dados_arquivo(self, nome_arquivo_limpo: str, gdf_dados: gpd.GeoDataFrame, metricas: dict, gdf_linhas: gpd.GeoDataFrame):
        """Adiciona layers de um arquivo específico ao mapa existente."""
        if not self.mapa:
            raise Exception("Mapa não inicializado. Chame iniciar_mapa() primeiro.")

        print(f"Adicionando layers para: {nome_arquivo_limpo}")

        # Preparação dos dados em WGS84
        # Revertido simplificação para manter detalhes completos (pedido do usuário)
        
        gdf_heatmap = gpd.GeoDataFrame({'valor': metricas['heatmap_valores'], 'geometry': metricas['heatmap_geometria']}, crs=gdf_linhas.crs).to_crs(epsg=4326)
        gdf_cotas = gpd.GeoDataFrame({'valor': metricas['cotas_dados'], 'geometry': metricas['cotas_geometria']}, crs=gdf_linhas.crs).to_crs(epsg=4326)
        gdf_linhas_wgs = gdf_linhas.to_crs(epsg=4326)
        
        # Sub-amostragem de pontos para visualização
        gdf_pontos_total = gdf_dados.to_crs(epsg=4326)
        if len(gdf_pontos_total) > 2000: # Limite por arquivo para não pesar
            gdf_pontos_total = gdf_pontos_total.sample(2000)

        # Atualiza bounds gerais
        self.bounds_geral.append(gdf_pontos_total.total_bounds)

        # Prefixo para agrupar layers visualmente (ex: [Arquivo1] Heatmap)
        prefix = f"<b>[{nome_arquivo_limpo}]</b>"
        
        # PASSO para cotas OK (filtra somente 1 a cada X)
        passo_vis = self.config.PASSOS_VISUALIZACAO[0] if self.config.PASSOS_VISUALIZACAO else 10

        # --- HEATMAP (GeoJSON Otimizado) ---
        if self.config.LAYER_VISIBILITY['heatmap']:
            fg_heat = folium.FeatureGroup(name=f"{prefix} {self.config.LAYER_NAMES['heatmap']}", show=True)
            
            # Constantes para estilo
            tol_min = self.config.TOLERANCIA_MIN
            tol_max = self.config.TOLERANCIA_MAX
            cor_ok = self.config.ESTILO['cor_cota_ok']
            cor_abaixo = self.config.ESTILO['cor_cota_abaixo']
            cor_acima = self.config.ESTILO['cor_cota_acima']
            weight = self.config.ESTILO['heatmap_espessura']
            opacity = self.config.ESTILO['heatmap_opacidade']

            def style_heatmap(feature):
                val = feature['properties']['valor']
                cor = cor_ok
                if val < tol_min: cor = cor_abaixo
                elif val > tol_max: cor = cor_acima
                return {
                    'color': cor,
                    'weight': weight,
                    'opacity': opacity
                }

            folium.GeoJson(
                gdf_heatmap,
                name=f"{prefix} Heatmap GeoJSON",
                style_function=style_heatmap
            ).add_to(fg_heat)
            
            fg_heat.add_to(self.mapa)

        # --- COTAS (Com Lógica de Zoom) ---
        # Separamos em Grupos OK (Zoom dependent) e Alertas (Sempre visible)
        fg_cotas = folium.FeatureGroup(name=f"{prefix} Cotas", show=self.config.LAYER_VISIBILITY['cotas_alerta'])
        
        for i, row in gdf_cotas.iterrows():
            val = row['valor']['val']
            is_alerta = val < self.config.TOLERANCIA_MIN or val > self.config.TOLERANCIA_MAX
            
            # SE PULAR COTAS OK: Se não for alerta, e config diz pra não mostrar, pule.
            if not is_alerta and not self.config.LAYER_VISIBILITY['cotas_ok']:
                continue
            
            # (Código de amostragem removido pois agora só mostramos alertas)
            
            cor = self.config.ESTILO['cor_cota_ok']
            if val < self.config.TOLERANCIA_MIN:
                cor = self.config.ESTILO['cor_cota_abaixo']
            elif val > self.config.TOLERANCIA_MAX:
                cor = self.config.ESTILO['cor_cota_acima']
            
            # Arredonda coordenadas para 6 casas decimais para reduzir tamanho do HTML
            coords = [(round(pt[1], 6), round(pt[0], 6)) for pt in row.geometry.coords] # LatLon
            
            
            # Adiciona Linha da Cota
            folium.PolyLine(coords, color=cor, weight=1, opacity=0.9).add_to(fg_cotas)
            
            # Lógica de Classes CSS para Zoom
            # cota-alert: sempre visivel (se layer ativa)
            # cota-lvl-1 (idx%20==0): Zoom Baixo
            # cota-lvl-2 (idx%5==0): Zoom Médio
            # cota-lvl-3 (resto): Zoom Alto
            zoom_class = "cota-alert" if is_alerta else "cota-lvl-3"
            if not is_alerta:
                if i % 20 == 0: zoom_class = "cota-lvl-1"
                elif i % 5 == 0: zoom_class = "cota-lvl-2"
            
            # Marcador
            centro_lat = (coords[0][0] + coords[1][0])/2
            centro_lon = (coords[0][1] + coords[1][1])/2
            
            # OTIMIZAÇÃO CSS: Usar classes em vez de style inline
            # Classes: cota-base, cota-fundo-{true/false}, cota-{ok/abaixo/acima}
            
            tipo_cota = "ok"
            if val < self.config.TOLERANCIA_MIN: tipo_cota = "abaixo"
            elif val > self.config.TOLERANCIA_MAX: tipo_cota = "acima"
            
            fundo_cls = "cota-fundo-true" if self.config.FUNDO_COTAS else "cota-fundo-false"
            
            # Estilos dinâmicos que precisam ser inline (apenas cor)
            style_inline = f"color: {cor};"
            if self.config.FUNDO_COTAS:
                bg_color = 'rgba(255,255,255,0.7)' if not is_alerta else cor
                text_color = 'black' if not is_alerta else 'white'
                border_color = cor
                style_inline = f"background: {bg_color}; color: {text_color}; border-color: {border_color};"
            
            # HTML minimalista
            html=f'<div class="cota-base {fundo_cls} cota-{tipo_cota}" style="{style_inline}">{val:.2f}m</div>'

            folium.map.Marker(
                [centro_lat, centro_lon],
                icon=folium.DivIcon(
                    icon_size=(0,0),
                    icon_anchor=(0,0),
                    class_name=f"cota-marker-container {zoom_class}", # Classe CSS aplicada aqui
                    html=html
                )
            ).add_to(fg_cotas)
            
        fg_cotas.add_to(self.mapa)

        # --- LINHAS TRAJETÓRIA (GeoJSON Otimizado) ---
        if self.config.LAYER_VISIBILITY['linhas']:
            fg_linhas = folium.FeatureGroup(name=f"{prefix} {self.config.LAYER_NAMES['linhas']}", show=False)
            
            cor_linha = self.config.ESTILO['cor_linha']
            weight_linha = self.config.ESTILO['linha_espessura']
            opacity_linha = self.config.ESTILO['linha_opacidade']

            folium.GeoJson(
                gdf_linhas_wgs,
                name=f"{prefix} Linhas GeoJSON",
                style_function=lambda feature: {
                    'color': cor_linha,
                    'weight': weight_linha,
                    'opacity': opacity_linha
                }
            ).add_to(fg_linhas)
            
            fg_linhas.add_to(self.mapa)

        # --- CLUSTERS LABELS ---
        if 'cluster_label' in gdf_linhas_wgs.columns:
            clusters = gdf_linhas_wgs.dissolve(by='cluster_label')
            for label, row in clusters.iterrows():
                centro = row.geometry.centroid
                folium.Marker(
                    [centro.y, centro.x],
                    icon=folium.DivIcon(
                        class_name="cluster-marker-container",
                        html=f"<div style='font-size: 24px; color: white; font-weight: bold; white-space: nowrap; pointer-events: none; text-shadow: 0 0 3px #000, 0 0 6px #000, 0 0 10px #000;'>{label}</div>"
                    )
                ).add_to(self.mapa)

    def salvar_mapa(self, pasta_saida: str, nome_base: str, html_painel_global: str):
        """Finaliza o mapa, injeta CSS/JS e salva."""
        if not self.mapa: return

        # Cálculo de Bounds Global
        bounds_projeto = None
        if self.bounds_geral:
            min_x = min(b[0] for b in self.bounds_geral)
            min_y = min(b[1] for b in self.bounds_geral)
            max_x = max(b[2] for b in self.bounds_geral)
            max_y = max(b[3] for b in self.bounds_geral)
            
            # Margem de 10%
            margem_x = (max_x - min_x) * 0.1
            margem_y = (max_y - min_y) * 0.1
            
            bounds_projeto = [[min_y - margem_y, min_x - margem_x], [max_y + margem_y, max_x + margem_x]]
            self.mapa.fit_bounds(bounds_projeto)
        
        # Adiciona TileLayer AGORA com bounds (Recorte de background)
        # max_native_zoom=20 permite zoom digital além do nível 20
        # control=False esconde do Layer Control
        # (O TileLayer já foi adicionado em iniciar_mapa para garantir carregamento, mas aqui ajustamos bounds se necessário)
        # folium.TileLayer(...).add_to(self.mapa) # REMOVIDO para evitar duplicata

        # 1. CSS (Zoom Classes + Painel Minimzável + Ícone Régua)
        css = """
        <style>
            /* Reset básico para garantir full screen */
            body, html, .folium-map { width: 100%; height: 100%; margin: 0; padding: 0; }
            
            .info-panel {
                position: fixed; bottom: 2px; left: 50%; transform: translateX(-50%);
                z-index: 9999; background: rgba(255,255,255,0.95); padding: 0;
                border: 1px solid #ccc; /* Borda leve */
                border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                min-width: 400px; 
                max-width: 95vw; max-height: 80vh; /* Aumentado para caber layers */
                transition: height 0.3s ease;
                display: flex; flex-direction: column; overflow: hidden;
                font-size: 12px;
            }
            .info-panel.minimized .panel-content {
                display: none !important;
            }
            /* Layers sempre visíveis, mesmo minimizado */
            .layers-container {
                border-bottom: 1px solid #ddd;
                background: #f9f9f9;
                padding: 3px;
                max-height: 30vh;
                overflow-y: auto;
            }
            .panel-header {
                display: flex; justify-content: center; align-items: center; position: relative;
                cursor: pointer; background: #f0f0f0; padding: 6px 10px; border-bottom: 1px solid #ccc;
                flex-shrink: 0;
            }
            .panel-title { font-weight: bold; font-size: 12px; text-align: center; }
            .panel-content {
                overflow-y: auto; padding: 6px; flex-grow: 1;
            }
            .stats-grid {
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin-bottom: 6px;
            }
            .stats-block {
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .stats-block:last-child {
                border-bottom: none;
            }
            .stats-grid > div {
                min-width: 0;
            }
            .toggle-btn {
                background: #ddd; border: none; padding: 2px 8px; border-radius: 4px; font-weight: bold; position: absolute; right: 8px;
            }
            .leaflet-control-measure-toggle {
                background-image: none !important; display: flex; align-items: center; justify-content: center;
                font-size: 20px; color: #000;
            }
            .leaflet-control-measure-toggle::after { content: '📏'; }
            .leaflet-control-attribution { display: none !important; }
            
            /* Lógica de Zoom para Cotas */
            .cota-lvl-2, .cota-lvl-3 { display: none; }
            .show-details .cota-lvl-2, .show-details .cota-lvl-3 { display: block !important; }
            .hide-all-cotas .cota-marker-container { display: none !important; }
            .cota-alert { display: block !important; }
            
            /* CLUSTERS: Visível apenas Zoom <= 17 */
            .hide-clusters .cluster-marker-container { display: none !important; }

            /* Layer Control Customization */
            .leaflet-control-layers {
                box-shadow: none !important; border: none !important; background: none !important;
                margin: 0 !important; padding: 0 !important; width: 100% !important;
            }
            .leaflet-control-layers-list { margin-bottom: 0; }
            .leaflet-control-layers-overlays {
                 display: block; width: 100%;
            }
            .leaflet-control-layers-overlays .layer-row {
                 display: flex; align-items: center; flex-wrap: wrap; gap: 6px 12px;
                 width: 100%; padding: 2px 0; border-bottom: 1px solid #eee;
            }
            .leaflet-control-layers-overlays .layer-row:last-child { border-bottom: none; }
            .leaflet-control-layers-overlays label {
                 box-sizing: border-box; margin: 0; font-size: 12px; width: auto;
                 display: inline-flex; align-items: center; background: transparent; padding: 2px 0; border-radius: 4px; border: none;
            }
            .cluster-marker-container {
                white-space: nowrap !important;
                pointer-events: none !important;
                text-shadow: 0 0 3px #000, 0 0 6px #000, 0 0 10px #000 !important;
            }
            
            /* Classes Otimizadas para Cotas (Reduz HTML size) */
            .cota-base {
                font-weight: bold; padding: 2px; border-radius: 4px;
                display: inline-block; pointer-events: none;
                transform: translate(-50%, -50%);
            }
            .cota-fundo-true {
                border: 1px solid;
            }
            .cota-fundo-false {
                background: none !important;
                text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;
            }
            /* Caso especial para amarelo (Acima) sem fundo: contorno preto */
            .cota-fundo-false.cota-acima {
                text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000 !important;
            }
        </style>
        """

        # 2. JS (Zoom + Toggle Panel + Zoom Display)
        map_id = self.mapa.get_name()
        
        # Ajuste: Define variáveis para injeção no JS
        min_zoom = self.config.ZOOM_MIN_COTAS
        max_cluster_zoom = self.config.ZOOM_MAX_CLUSTERS
        
        js = f"""
        <script>
            function togglePanel() {{
                var panel = document.querySelector('.info-panel');
                panel.classList.toggle('minimized');
                var btn = document.querySelector('.toggle-btn');
                btn.innerText = panel.classList.contains('minimized') ? '+' : '-';
            }}

            document.addEventListener("DOMContentLoaded", function() {{
                var map = {map_id};
                var minDetailZoom = {min_zoom};
                var maxClusterZoom = {max_cluster_zoom};
                
                map.dragging.enable();
                map.scrollWheelZoom.enable();
                map.touchZoom.enable();
                
                // --- MOSTRADOR DE ZOOM ---
                var zoomDisplay = document.createElement('div');
                zoomDisplay.id = 'zoom-display';
                zoomDisplay.innerHTML = 'Zoom Atual: <span id="z-val">--</span>';
                zoomDisplay.style.cssText = 'position:fixed; top:10px; left:50%; transform:translateX(-50%); background:rgba(0,0,0,0.8); color:white; padding:5px 15px; border-radius:20px; font-size:14px; font-weight:bold; z-index:10000; pointer-events:none; box-shadow: 0 2px 5px rgba(0,0,0,0.5);';
                document.body.appendChild(zoomDisplay);

                function updateZoomText() {{
                    var z = map.getZoom();
                    var el = document.getElementById('z-val');
                    if(el) el.innerText = z;
                }}

                // --- LOGICA DE CAMADAS ---
                
                // Mapa de referências para as layers (Nome -> Objeto Layer)
                var layerRef = {{}};

                setTimeout(function() {{
                    var layerControlInstance = null;
                    for (var key in window) {{
                        try {{
                            if (window[key] instanceof L.Control.Layers) {{
                                layerControlInstance = window[key];
                                break;
                            }}
                        }} catch(e) {{}}
                    }}

                    if (layerControlInstance && layerControlInstance._layers) {{
                        layerControlInstance._layers.forEach(function(obj) {{
                            if (obj.name) {{
                                layerRef[obj.name.trim()] = obj.layer;
                            }}
                        }});
                    }}

                    map.eachLayer(function(layer) {{
                        if (layer.options && layer.options.name) {{
                            layerRef[layer.options.name.trim()] = layer;
                        }}
                    }});

                    var layerControl = document.querySelector('.leaflet-control-layers');
                    var layersContainer = document.querySelector('.layers-container');
                    
                    if (layerControl && layersContainer) {{
                        layersContainer.innerHTML = '';
                        layersContainer.appendChild(layerControl);
                        layerControl.style.display = '';

                        var overlaysContainer = layerControl.querySelector('.leaflet-control-layers-overlays');
                        var overlayLabels = layerControl.querySelectorAll('.leaflet-control-layers-overlays label');
                        var trendByGroup = {{}};
                        var cotasByGroup = {{}};

                        if (overlaysContainer) {{
                            var groups = {{}};
                            overlayLabels.forEach(function(label) {{
                                var span = label.querySelector('span');
                                if (!span) return;
                                var text = span.textContent || '';
                                var match = text.match(/\\[(.*?)\\]/);
                                var group = match ? match[1] : 'Outros';
                                if (!groups[group]) groups[group] = [];
                                groups[group].push(label);
                            }});

                            overlaysContainer.innerHTML = '';

                            Object.keys(groups).forEach(function(groupName) {{
                                var row = document.createElement('div');
                                row.className = 'layer-row';

                                groups[groupName].forEach(function(label) {{
                                    row.appendChild(label);
                                }});

                                overlaysContainer.appendChild(row);
                            }});
                        }}

                        overlayLabels.forEach(function(label) {{
                            var span = label.querySelector('span');
                            var input = label.querySelector('input');
                            if (!span || !input) return;
                            var text = span.textContent || '';
                            var match = text.match(/\\[(.*?)\\]/);
                            if (!match) return;
                            var group = match[1];
                            if (text.indexOf('Mapa de Tendência') !== -1) {{
                                trendByGroup[group] = input;
                            }}
                            if (text.indexOf('Cotas') !== -1) {{
                                cotasByGroup[group] = input;
                            }}
                        }});

                        function linkTrendToCotas(trendInput, cotasInput) {{
                            if (!trendInput || !cotasInput) return;
                            trendInput.addEventListener('change', function() {{
                                if (!this.checked && cotasInput.checked) {{
                                    cotasInput.checked = false;
                                    cotasInput.dispatchEvent(new Event('change'));
                                }}
                            }});
                        }}

                        for (var g in trendByGroup) {{
                            linkTrendToCotas(trendByGroup[g], cotasByGroup[g]);
                        }}
                    }}

                    // Removemos a lógica antiga de "Reforço de Eventos" pois agora controlamos diretamente
                    console.log("Painel customizado carregado. Total layers:", Object.keys(layerRef).length);
                }}, 800);

                function updateVisibility() {{
                    var z = map.getZoom();
                    updateZoomText();
                    
                    var container = map.getContainer();
                    
                    // Lógica Cotas
                    if (z >= minDetailZoom) {{
                        container.classList.add('show-details');
                        container.classList.remove('hide-all-cotas');
                    }} else {{
                        container.classList.add('hide-all-cotas');
                        container.classList.remove('show-details');
                    }}
                    
                    // Lógica Clusters
                    // Se cotas aparecem, clusters somem. Sincronizado.
                    if (z >= minDetailZoom) {{
                        container.classList.add('hide-clusters');
                    }} else {{
                        container.classList.remove('hide-clusters');
                    }}
                }}
                
                map.on('zoomend', updateVisibility);
                map.on('moveend', updateVisibility); // Reforço
                updateVisibility();
            }});
        </script>
        """
        
        # Painel HTML Estruturado
        painel_div = f'''
        <div class="info-panel">
            <div class="panel-header" onclick="togglePanel()">
                <div class="panel-title">Informações</div>
                <button class="toggle-btn">-</button>
            </div>
            
            <!-- Legenda de Cores -->
            <div style="display: flex; justify-content: center; align-items: center; gap: 15px; padding: 5px; background: #fafafa; border-bottom: 1px solid #ddd; font-size: 11px;">
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 12px; height: 12px; background: {self.config.ESTILO['cor_cota_ok']}; border: 1px solid #999; border-radius: 2px;"></div>
                    <span>OK</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 12px; height: 12px; background: {self.config.ESTILO['cor_cota_abaixo']}; border: 1px solid #999; border-radius: 2px;"></div>
                    <span>Abaixo</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 12px; height: 12px; background: {self.config.ESTILO['cor_cota_acima']}; border: 1px solid #999; border-radius: 2px;"></div>
                    <span>Acima</span>
                </div>
            </div>
            
            <!-- Area de Layers Fixa (Não some no minimize) -->
            <div class="layers-container">
                <!-- Layers injetados aqui via JS -->
            </div>

            <!-- Conteudo Minimizavel -->
            <div class="panel-content">
                <div class="stats-grid">
                    {html_painel_global}
                </div>
            </div>
        </div>
        '''
        
        self.mapa.get_root().html.add_child(folium.Element(css))
        self.mapa.get_root().html.add_child(folium.Element(js))
        self.mapa.get_root().html.add_child(folium.Element(painel_div))

        # MeasureControl removido conforme solicitado
        # MeasureControl(position='topright', primary_length_unit='meters', active_color='#00FF00').add_to(self.mapa)
        # Fullscreen().add_to(self.mapa)
        folium.LayerControl(collapsed=False).add_to(self.mapa)

        caminho_final = os.path.join(pasta_saida, f"{nome_base}.html")
        self.mapa.save(caminho_final)
        print(f"Mapa salvo em: {caminho_final}")

# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    # Garante saída em UTF-8 no terminal Windows para evitar "mojibake"
    # Usa reconfigure (Python 3.7+) para não quebrar o buffer/flush
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            # Fallback para versões muito antigas (embora o user use 3.13)
            pass

    inicio_proc = datetime.now()
    dir_atual = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Configuração e Inicialização
    arquivos_zips = GerenciadorArquivos.buscar_todos_zips(dir_atual)
    if not arquivos_zips:
        print("Nenhum arquivo ZIP encontrado neste diretório.")
        return

    print(f"Arquivos encontrados: {len(arquivos_zips)}")
    
    viz_svg = VisualizadorMapa(TABELA_CONFIG)
    viz_canvas = VisualizadorMapa(TABELA_CONFIG)
    viz_svg.iniciar_mapa(prefer_canvas=False)
    viz_canvas.iniciar_mapa(prefer_canvas=True)
    
    html_painel_agregado = ""
    stats_globais = {'tiro_medio_soma': 0, 'tiro_medio_conta': 0, 'espacamento_soma': 0, 'espacamento_conta': 0}
    
    pasta_extraida_base = os.path.join(dir_atual, "dados_extraidos")

    # 2. Processamento Loop
    for zip_path in arquivos_zips:
        nome_arquivo_full = os.path.basename(zip_path)
        nome_arquivo_limpo = os.path.splitext(nome_arquivo_full)[0] # Remove .zip
        print(f"\n>>> Processando: {nome_arquivo_limpo}")
        
        try:
            # Extração Única
            pasta_dados = GerenciadorArquivos.extrair_zip(zip_path, pasta_extraida_base)
            shp_path = GerenciadorArquivos.encontrar_shapefile(pasta_dados)
            
            if not shp_path:
                print("  [X] Shapefile não encontrado no ZIP.")
                continue
                
            # Carga
            gdf_bruto = gpd.read_file(shp_path)
            if gdf_bruto.empty: continue
            
            # Projeção
            if not gdf_bruto.crs.is_projected:
                centro = gdf_bruto.geometry.iloc[0]
                epsg = UtilitariosGeo.calcular_epsg_utm(centro.y, centro.x)
                gdf_utm = gdf_bruto.to_crs(epsg=epsg)
            else:
                gdf_utm = gdf_bruto
                
            # Processamento
            linhas_geom, metadados_linhas = ProcessadorTrajetoria.gerar_linhas_de_voo(gdf_utm, TABELA_CONFIG)
            if not linhas_geom: 
                print("  [!] Falha ao gerar linhas.")
                continue
                
            gdf_linhas = gpd.GeoDataFrame(metadados_linhas, geometry=linhas_geom, crs=gdf_utm.crs)
            
            # Clustering e União
            gdf_linhas = ProcessadorTrajetoria.agrupar_por_clusters(gdf_linhas, TABELA_CONFIG)
            
            # ATUALIZAÇÃO: Prefixo no Cluster (ex: "Arquivo A")
            if 'cluster_label' in gdf_linhas.columns:
                gdf_linhas['cluster_label'] = gdf_linhas['cluster_label'].apply(lambda l: f"{nome_arquivo_limpo} {l}")

            gdf_linhas = ProcessadorTrajetoria.conectar_segmentos_quebrados(gdf_linhas, TABELA_CONFIG)
            
            # Análise
            analisador = AnalisadorEspacamento(TABELA_CONFIG)
            metricas = analisador.processar(gdf_linhas)
            
            # Relatórios
            print("  --- stats ---")
            media, tiro_medio = GeradorRelatorio.exibir_terminal(metricas, gdf_linhas, TABELA_CONFIG)
            
            # Acumula Stats Globais
            if media > 0:
                stats_globais['espacamento_soma'] += media
                stats_globais['espacamento_conta'] += 1
            if tiro_medio > 0:
                stats_globais['tiro_medio_soma'] += tiro_medio
                stats_globais['tiro_medio_conta'] += 1

            # HTML Parcial (Com título do arquivo) - Wrapped em div para grid
            html_painel = GeradorRelatorio.gerar_html_tabela(media, tiro_medio, gdf_linhas, TABELA_CONFIG)
            html_painel_agregado += f"<div><h4 style='margin:0 0 5px 0; text-align:center; font-weight:bold; font-size:12px;'>{nome_arquivo_limpo}</h4>{html_painel}</div>"
            
            # Adiciona ao Mapa
            viz_svg.adicionar_dados_arquivo(nome_arquivo_limpo, gdf_utm, metricas, gdf_linhas)
            viz_canvas.adicionar_dados_arquivo(nome_arquivo_limpo, gdf_utm, metricas, gdf_linhas)
            
        except Exception as e:
            print(f"  [ERRO CRÍTICO] Falha ao processar {nome_arquivo_limpo}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Finalização
    print("\n==========================================")
    print("Processamento concluído.")
    
    primeiro_nome = os.path.splitext(os.path.basename(arquivos_zips[0]))[0]
    codigo_base = primeiro_nome.split('_')[0] if '_' in primeiro_nome else primeiro_nome
    nome_base = f"Mapa Espaçamento Plantio - {codigo_base}"
    pasta_mapas = os.path.join(dir_atual, "mapas")
    os.makedirs(pasta_mapas, exist_ok=True)
    
    # Salva Mapa
    viz_svg.salvar_mapa(pasta_mapas, f"{nome_base} - svg", html_painel_agregado)
    viz_canvas.salvar_mapa(pasta_mapas, f"{nome_base} - canvas", html_painel_agregado)

    tempo_total = datetime.now() - inicio_proc
    print(f"Tempo total: {str(tempo_total).split('.')[0]}")

if __name__ == "__main__":
    main()
