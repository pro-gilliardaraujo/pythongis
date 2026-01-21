import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import substring
import math
import os
import glob
import zipfile
import folium
from folium.plugins import MeasureControl, Fullscreen
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAÇÕES GERAIS - Ajuste aqui os parâmetros de análise e visualização
# ==============================================================================
CONFIG = {
    # Visualização do Mapa
    'MAPA_ZOOM_INICIAL': 18,
    'MAPA_MAX_ZOOM': 24, # Permite zoom muito alto
    'MAPA_LIMITAR_VISAO_AO_PROJETO': True, # Restringe a navegação à área do projeto (remove "background infinito")
    
    # Configuração de Camadas (Layers)
    'GERAR_CAMADAS_OCULTAS': False, # OTIMIZAÇÃO: Se False, não gera dados para camadas marcadas como False (Reduz tamanho do arquivo)
    'LAYER_NAME_PONTOS': 'Pontos (Original)',
    'LAYER_SHOW_PONTOS': False, 
    'LAYER_NAME_LINHAS': 'Linhas (Calculadas)',
    'LAYER_SHOW_LINHAS': False,
    'LAYER_NAME_COTAS_OK': 'Cotas (OK)',
    'LAYER_SHOW_COTAS_OK': False,
    'LAYER_NAME_COTAS_ALERTA': 'Cotas (Fora)',
    'LAYER_SHOW_COTAS_ALERTA': True,
    'LAYER_NAME_HEATMAP': 'Mapa de Tendência',
    'LAYER_SHOW_HEATMAP': True,
    
    # Estilo dos Pontos
    'PONTOS_RAIO': 1,
    'PONTOS_COR_PADRAO': '#FF4500', # OrangeRed
    'PONTOS_COR_ATIVO': '#00FF00', # Lime
    'PONTOS_COR_INATIVO': '#FF0000', # Red
    
    # Estilo das Linhas Geradas (Passadas)
    'LINHAS_COR': '#00FFFF', # Cyan
    'LINHAS_ESPESSURA': 3,
    'LINHAS_OPACIDADE': 0.6,
    
    # Estilo das Cotas Automáticas (Medições)
    'LISTA_PASSOS_VISUALIZACAO': [10], # Gera mapas para cada passo (1 = Todas as cotas)
    'COTA_COR_OK': '#00FF00', # Verde
    'COTA_COR_ALERTA': '#FF0000', # Vermelho
    'COTA_ESPESSURA': 2,
    
    # Estilo Mapa de Calor
    'HEATMAP_ESPESSURA': 7, # Linha grossa
    'HEATMAP_OPACIDADE': 0.7,
    
    # Parâmetros de Tolerância (O que é aceitável?)
    'TOLERANCIA_MIN': 2.9, # Abaixo disso é alerta
    'TOLERANCIA_MAX': 3.1, # Acima disso é alerta
    'FATOR_CORTE_VISUALIZACAO': 2.0, # Multiplicador sobre a TOLERANCIA_MAX. Se dist > (MAX * 2), a cota é descartada (gap entre talhões).
    
    # Parâmetros de Geração de Linha
    'QUEBRA_TEMPO_SEC': 60,
    'QUEBRA_ANGULO_GRAUS': 60,
    
    # Parâmetros de Cálculo de Espaçamento
    # NOTA: O filtro de amostragem e offset é feito no Python.
    'OFFSET_CABECEIRA_METROS': 10.0,
    'AMOSTRAGEM_DIST_METROS': 5.0,
    'MIN_DIST_ENTRE_COTAS_MESMA_LINHA': 3.0, # Distância mínima entre cotas na mesma linha (limpeza visual)
    'FILTRO_DIST_MIN': 1.35, # Aumentado para ignorar sobreposições e segmentos colineares muito próximos
    'FILTRO_DIST_MAX': 15.0, # Reduzido para evitar conectar com linhas muito distantes (ex: 24m)
    'MIN_DIST_TOPOLOGICA': 50.0, # Distância mínima AO LONGO DA LINHA para considerar como vizinho (evita auto-interseção local)
}

def calcular_utm_epsg(lat, lon):
    """
    Calcula o código EPSG da zona UTM para uma dada latitude e longitude.
    Isso é necessário para converter coordenadas geográficas (graus) em métricas (metros),
    permitindo cálculos precisos de distância.
    
    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        
    Returns:
        int: Código EPSG (ex: 32722 para zona 22S).
    """
    # A Terra é dividida em zonas de 6 graus.
    zone = math.floor((lon + 180) / 6) + 1
    
    # Define se é Hemisfério Norte (326xx) ou Sul (327xx)
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return int(epsg)

def localizar_zip_na_pasta(pasta_base):
    """
    Procura o primeiro arquivo .zip na pasta base.
    Se existirem vários, pega o mais recente pela data de modificação.
    
    Args:
        pasta_base (str): Diretório raiz do projeto.
        
    Returns:
        str ou None: Caminho do zip encontrado ou None se não existir.
    """
    arquivos_zip = glob.glob(os.path.join(pasta_base, "*.zip"))
    if not arquivos_zip:
        return None
    arquivos_zip.sort(key=os.path.getmtime, reverse=True)
    return arquivos_zip[0]

def extrair_zip(caminho_zip, pasta_saida):
    """
    Extrai o conteúdo do arquivo .zip para uma pasta de saída.
    
    Args:
        caminho_zip (str): Caminho do arquivo zip.
        pasta_saida (str): Pasta onde o conteúdo será extraído.
        
    Returns:
        str: Caminho absoluto da pasta extraída.
    """
    os.makedirs(pasta_saida, exist_ok=True)
    with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
        zip_ref.extractall(pasta_saida)
    return pasta_saida

def gerar_segmentos_por_tempo_direcao(gdf_pontos_utm, coluna_tempo='Time'):
    """
    Gera segmentos de linha (listas de índices) a partir de pontos usando o tempo e a direção aproximada.
    A ideia é separar passadas diferentes quando há:
    - Grande salto de tempo.
    - Mudança brusca de direção.
    
    Args:
        gdf_pontos_utm (GeoDataFrame): Pontos em coordenadas UTM (metros).
        coluna_tempo (str): Nome da coluna de tempo no shapefile.
        
    Returns:
        list: Lista de segmentos com índices originais dos pontos.
    """
    if coluna_tempo not in gdf_pontos_utm.columns:
        return []
    
    pontos = gdf_pontos_utm.copy()
    pontos[coluna_tempo] = pd.to_datetime(
        pontos[coluna_tempo],
        format='%m/%d/%Y %I:%M:%S %p',
        errors='coerce'
    )
    pontos = pontos.dropna(subset=[coluna_tempo])
    pontos = pontos.sort_values(coluna_tempo).reset_index(drop=True)
    
    if len(pontos) < 3:
        return []
    
    segmentos = []
    segmento_atual = []
    
    tempo_anterior = None
    angulo_anterior = None
    
    for i in range(1, len(pontos)):
        p1 = pontos.geometry.iloc[i - 1]
        p2 = pontos.geometry.iloc[i]
        
        dt = (pontos[coluna_tempo].iloc[i] - pontos[coluna_tempo].iloc[i - 1]).total_seconds()
        
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        angulo = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        
        if tempo_anterior is None:
            tempo_anterior = dt
        
        if angulo_anterior is None:
            angulo_anterior = angulo
        
        variacao_angulo = abs(angulo - angulo_anterior)
        variacao_angulo = min(variacao_angulo, 360 - variacao_angulo)
        
        if dt > CONFIG['QUEBRA_TEMPO_SEC'] or variacao_angulo > CONFIG['QUEBRA_ANGULO_GRAUS']:
            if len(segmento_atual) >= 3:
                segmentos.append(segmento_atual)
            segmento_atual = []
        
        segmento_atual.append(pontos['orig_index'].iloc[i])
        
        tempo_anterior = dt
        angulo_anterior = angulo
    
    if len(segmento_atual) >= 3:
        segmentos.append(segmento_atual)
    
    return segmentos

def analisar_espacamento_e_gerar_mapa(caminho_shapefile, pasta_saida_mapas):
    """
    Função principal que:
    1. Carrega o arquivo Shapefile.
    2. Identifica o sistema de coordenadas e converte para UTM (metros).
    3. Analisa o espaçamento entre pontos/linhas.
    4. Gera um mapa interativo HTML com os dados.
    
    Args:
        caminho_shapefile (str): Caminho absoluto ou relativo para o arquivo .shp.
    """
    print(f"Carregando arquivo: {caminho_shapefile}")
    
    # Tenta ler o arquivo usando Geopandas
    try:
        gdf = gpd.read_file(caminho_shapefile)
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return

    if gdf.empty:
        print("O arquivo shapefile está vazio.")
        return

    print(f"Carregados {len(gdf)} elementos.")

    # Calcula nome base do arquivo (Data de Ontem) para uso no Título e Nome do Arquivo
    ontem = datetime.now() - timedelta(days=1)
    data_str = ontem.strftime('%d-%m-%Y')
    nome_base_titulo = f"Mapa Espaçamento Plantio - {data_str}"

    # ---------------------------------------------------------
    # 1. CONVERSÃO DE COORDENADAS (PROJEÇÃO)
    # ---------------------------------------------------------
    # Verifica se o sistema de coordenadas (CRS) é projetado (em metros) ou geográfico (graus).
    # Se for geográfico, precisamos converter para UTM para medir distâncias em metros.
    if not gdf.crs.is_projected:
        print("Sistema de coordenadas é geográfico (Lat/Lon). Convertendo para UTM para cálculos em metros...")
        
        # Pega os limites (bounds) para achar o centro do projeto
        minx, miny, maxx, maxy = gdf.total_bounds
        centro_lon = (minx + maxx) / 2
        centro_lat = (miny + maxy) / 2
        
        # Calcula a zona UTM correta automaticamente
        target_epsg = calcular_utm_epsg(centro_lat, centro_lon)
        print(f"Zona UTM detectada automaticamente (EPSG): {target_epsg}")
        
        # Realiza a conversão
        gdf_utm = gdf.to_crs(epsg=target_epsg)
    else:
        print(f"O sistema já é projetado: {gdf.crs.name}")
        gdf_utm = gdf

    # ---------------------------------------------------------
    # 2. ANÁLISE DE METADADOS
    # ---------------------------------------------------------
    print("\n--- Metadados Encontrados ---")
    print("Tipos de Geometria:")
    print(gdf_utm.geometry.type.value_counts())
    
    # Se existir a coluna SWATHWIDTH (Largura da Faixa), mostra os valores
    if 'SWATHWIDTH' in gdf_utm.columns:
        contagem_swath = gdf_utm['SWATHWIDTH'].value_counts()
        print(f"\nColuna 'SWATHWIDTH' encontrada. Valores mais comuns (em metros): \n{contagem_swath}")
    
    # ---------------------------------------------------------
    # 3. CÁLCULO DE ESPAÇAMENTO (ESTIMATIVA)
    # ---------------------------------------------------------
    stats_html = f"<h3>{nome_base_titulo}</h3>"
    
    # Filtra apenas pontos para análise
    pontos = gdf_utm[gdf_utm.geometry.type.isin(['Point', 'MultiPoint'])].copy()
    pontos['orig_index'] = pontos.index
    
    linhas_geradas = []
    linhas_geradas_wgs = []
    
    if not pontos.empty:
        print("\n--- Iniciando Análise de Espaçamento ---")
        # Filtra pontos onde houve aplicação (se a coluna existir), para ignorar deslocamentos vazios
        if 'AppliedRate' in pontos.columns:
            pontos_ativos = pontos[pontos['AppliedRate'] > 0]
            if not pontos_ativos.empty:
                pontos = pontos_ativos
                print(f"Filtrado para pontos com 'AppliedRate > 0'. Total: {len(pontos)}")
            else:
                print("Nenhum ponto com 'AppliedRate > 0' encontrado. Usando todos os pontos.")

        print("Tentando gerar linhas a partir dos pontos (por tempo e direção)...")
        segmentos = gerar_segmentos_por_tempo_direcao(pontos)
        print(f"Segmentos gerados: {len(segmentos)}")
        
        if len(segmentos) > 0:
            pontos_utm_index = pontos.set_index('orig_index')
            pontos_wgs = gdf.loc[pontos_utm_index.index].copy()
            pontos_wgs['orig_index'] = pontos_wgs.index
            pontos_wgs = pontos_wgs.set_index('orig_index')
            
            for seg in segmentos:
                try:
                    linha_utm = LineString(pontos_utm_index.loc[seg].geometry.tolist())
                    linha_wgs = LineString(pontos_wgs.loc[seg].geometry.tolist())
                    linhas_geradas.append(linha_utm)
                    linhas_geradas_wgs.append(linha_wgs)
                except:
                    continue
        
        distancias = []
        
        cotas_geometry = []
        cotas_valores = []
        
        heatmap_segments_utm = []
        heatmap_values = []
        
        if len(linhas_geradas) > 1:
            print("Calculando distâncias LATERAIS (Swath) entre linhas geradas...")
            linhas_gdf = gpd.GeoDataFrame(geometry=linhas_geradas, crs=gdf_utm.crs)
            sindex = linhas_gdf.sindex
            
            # Para cada linha, amostramos pontos e medimos distância para a linha vizinha
            for idx, row in linhas_gdf.iterrows():
                geom = row.geometry
                length = geom.length
                
                # Variável para controle de densidade espacial na MESMA linha
                ultimo_pos_valida = -9999.0
                
                # Cria pontos de amostragem a cada X metros (ALTA DENSIDADE para o Frontend filtrar)
                # Exportamos metadados de posição para o JS poder aplicar o Offset dinamicamente
                
                # Gerar pontos a cada 1 metro
                distancias_amostragem = np.arange(0, length, CONFIG['AMOSTRAGEM_DIST_METROS'])
                if len(distancias_amostragem) == 0:
                     distancias_amostragem = [length / 2]
                
                pontos_amostra = [geom.interpolate(d) for d in distancias_amostragem]
                
                for i, pt in enumerate(pontos_amostra):
                    dist_from_start = distancias_amostragem[i]
                    if CONFIG['OFFSET_CABECEIRA_METROS'] > 0:
                        if dist_from_start < CONFIG['OFFSET_CABECEIRA_METROS'] or dist_from_start > (length - CONFIG['OFFSET_CABECEIRA_METROS']):
                            continue
                    
                    # Busca linhas candidatas num raio de busca (ex: 50m)
                    candidatos_idx = list(sindex.intersection(pt.buffer(CONFIG['FILTRO_DIST_MAX']).bounds))
                    candidatos = linhas_gdf.iloc[candidatos_idx]
                    
                    min_dist = float('inf')
                    found_neighbor = False
                    ponto_vizinho_mais_proximo = None
                    
                    for cand_idx, cand_row in candidatos.iterrows():
                        # Removido filtro de 'mesmo indice' para permitir vizinhos na mesma polilinha (curvas suaves)
                        # if cand_idx == idx: continue 
                        
                        # Para achar o ponto exato na linha vizinha, usamos nearest_points da shapely ops
                        from shapely.ops import nearest_points
                        p1, p2 = nearest_points(pt, cand_row.geometry)
                        
                        dist = p1.distance(p2)
                        
                        # Filtra distâncias muito pequenas (sobreposições ou mesmo segmento quebrado)
                        if dist < CONFIG['FILTRO_DIST_MIN']:
                            continue

                        # Se for a MESMA linha, verifica a distância topológica (ao longo do caminho)
                        # Se os pontos estiverem geometricamente perto (3m) mas topologicamente longe (100m), é uma passada paralela!
                        if cand_idx == idx:
                            # Projeta p2 na linha para saber a posição linear
                            pos2 = cand_row.geometry.project(p2)
                            dist_topologica = abs(pos2 - dist_from_start)
                            
                            if dist_topologica < CONFIG['MIN_DIST_TOPOLOGICA']:
                                continue # É o mesmo segmento local, ignora

                        if dist < min_dist:
                            # VALIDAÇÃO DE ÂNGULO: A cota deve ser perpendicular à linha original
                            # Calculamos o vetor da linha (tangente) e o vetor da cota
                            
                            # Pega um ponto um pouco a frente na linha para vetor tangente
                            delta = 1.0 # metros
                            pt_frente = geom.interpolate(dist_from_start + delta)
                            if dist_from_start + delta > length:
                                pt_frente = geom.interpolate(dist_from_start - delta)
                                vec_linha = (pt.x - pt_frente.x, pt.y - pt_frente.y)
                            else:
                                vec_linha = (pt_frente.x - pt.x, pt_frente.y - pt.y)
                            
                            # Vetor da cota
                            vec_cota = (p2.x - pt.x, p2.y - pt.y)
                            
                            # Produto Escalar para achar o ângulo
                            import math
                            def get_angle(v1, v2):
                                dot = v1[0]*v2[0] + v1[1]*v2[1]
                                mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                                mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                                if mag1 * mag2 == 0: return 0
                                cos_theta = dot / (mag1 * mag2)
                                cos_theta = max(min(cos_theta, 1), -1) # Clamp
                                return math.degrees(math.acos(cos_theta))
                            
                            angulo = get_angle(vec_linha, vec_cota)
                            
                            # Ângulo deve ser próximo de 90 graus (ex: entre 45 e 135)
                            # Se for perto de 0 ou 180, é longitudinal (paralelo)
                            if not (45 <= angulo <= 135):
                                continue 

                            min_dist = dist
                            found_neighbor = True
                            ponto_vizinho_mais_proximo = p2
                    
                    if found_neighbor:
                        # --- FILTRO 1: Descarte de distâncias muito grandes (gaps entre talhões) ---
                        limite_descarte = CONFIG['TOLERANCIA_MAX'] * CONFIG['FATOR_CORTE_VISUALIZACAO']
                        if min_dist > limite_descarte:
                            continue

                        # --- GERAÇÃO DO MAPA DE CALOR (HEATMAP) ---
                        # Gera um segmento da linha original colorido conforme a qualidade
                        # Usamos substring para pegar um pedaço da linha em torno do ponto de amostragem
                        half_sample = CONFIG['AMOSTRAGEM_DIST_METROS'] / 2
                        h_start = max(0, dist_from_start - half_sample)
                        h_end = min(length, dist_from_start + half_sample)
                        
                        try:
                            # Recorta o segmento da linha original
                            seg_geom = substring(geom, h_start, h_end)
                            heatmap_segments_utm.append(seg_geom)
                            heatmap_values.append(min_dist)
                        except Exception as e:
                            pass # Ignora erros de geometria pontuais

                        # --- FILTRO 2: Limpeza visual de cotas muito próximas na mesma linha ---
                        if (dist_from_start - ultimo_pos_valida) < CONFIG['MIN_DIST_ENTRE_COTAS_MESMA_LINHA']:
                             continue
                        
                        ultimo_pos_valida = dist_from_start # Atualiza a posição da última cota válida

                        distancias.append(min_dist)
                        # Salva a geometria da cota (Linha entre P1 e P2)
                        cotas_geometry.append(LineString([pt, ponto_vizinho_mais_proximo]))
                        cotas_valores.append({
                            'val': min_dist,
                            'pos': dist_from_start,     # Posição na linha (metros do início)
                            'len': length               # Comprimento total da linha pai
                        })

        else:
            print("Não foi possível gerar linhas suficientes para calcular espaçamento.")
        
        distancias = np.array(distancias)
        
        # Filtro básico de validade para o RELATÓRIO ESTATÍSTICO (apenas)
        # O Mapa receberá tudo.
        mask_validos = (distancias > 0.5) & (distancias < 50.0) # Valores fixos razoáveis para estatística
        distancias_validas = distancias[mask_validos]
        
        # Filtrando também as cotas
        # Para o JSON do mapa, vamos mandar TUDO que for geometricamente válido, o JS filtra visualmente.
        # Mas para não explodir o tamanho do arquivo, podemos cortar extremos absurdos (>100m)
        
        if len(distancias_validas) > 0:
            # ------------------------------------------------------------------
            # ESTATÍSTICA ROBUSTA (IQR - Intervalo Interquartil)
            # Remove outliers para calcular média/mediana reais
            # ------------------------------------------------------------------
            q1 = np.percentile(distancias_validas, 25)
            q3 = np.percentile(distancias_validas, 75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            
            distancias_robustas = distancias_validas[(distancias_validas >= limite_inferior) & (distancias_validas <= limite_superior)]
            
            if len(distancias_robustas) == 0:
                 distancias_robustas = distancias_validas # Fallback se filtrar tudo
            
            media = np.mean(distancias_robustas)
            mediana = np.median(distancias_robustas)
            minimo = np.min(distancias_validas) # Mínimo absoluto ainda é útil ver
            maximo = np.max(distancias_validas) # Máximo absoluto idem
            
            print(f"Média Estimada (Robusta): {media:.2f} m")
            print(f"Mediana Estimada (Robusta): {mediana:.2f} m")
            print(f"Mínimo Absoluto: {minimo:.2f} m")
            
            # CÁLCULO DO TIRO MÉDIO
            # O 'tiro' é o comprimento de cada segmento contínuo (linha gerada)
            comprimentos_tiros = [linha.length for linha in linhas_geradas]
            tiro_medio = np.mean(comprimentos_tiros) if comprimentos_tiros else 0
            
            # Adiciona ao HTML (SIMPLIFICADO)
            stats_html += f"""
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr><td><b>Espaçamento médio</b></td><td>{media:.2f} m</td></tr>
                <tr><td><b>Tiro médio</b></td><td>{tiro_medio:.2f} m</td></tr>
            </table>
            """
        else:
            print("Não foi possível determinar o espaçamento entre linhas (talvez seja passe único).")
            stats_html += "<p>Não foi possível calcular espaçamento entre linhas (possível passe único ou dados esparsos).</p>"
    
    # ---------------------------------------------------------
    # 4. GERAÇÃO DOS MAPAS INTERATIVOS (LOOP POR DENSIDADE)
    # ---------------------------------------------------------
    print("\n--- Gerando Mapas Interativos ---")
    
    # Prepara dados para o mapa (WGS84)
    gdf_mapa = gdf.copy()
    
    # Se tiver muitos pontos (> 5000), faz uma amostragem para o mapa não travar
    if len(gdf_mapa) > 5000:
        print(f"Reduzindo pontos para visualização (de {len(gdf_mapa)} para 5000)...")
        gdf_mapa = gdf_mapa.sample(5000)
    
    # Pega o centro do mapa
    centro_lat = gdf_mapa.geometry.y.mean()
    centro_lon = gdf_mapa.geometry.x.mean()
    
    # Calcula limites para restringir a visão (max_bounds)
    min_x, min_y, max_x, max_y = gdf_mapa.total_bounds
    # Adiciona uma margem de 10%
    margem_x = (max_x - min_x) * 0.1
    margem_y = (max_y - min_y) * 0.1
    bounds_projeto = [
        [min_y - margem_y, min_x - margem_x], # Sudoeste
        [max_y + margem_y, max_x + margem_x]  # Nordeste
    ]
    
    # Prepara cotas em WGS84 uma única vez
    cotas_gdf_wgs = None
    if len(cotas_geometry) > 0:
        cotas_gdf_utm = gpd.GeoDataFrame(
            {'valor': cotas_valores, 'geometry': cotas_geometry}, 
            crs=gdf_utm.crs
        )
        cotas_gdf_wgs = cotas_gdf_utm.to_crs(epsg=4326)

    # Prepara Heatmap em WGS84
    heatmap_gdf_wgs = None
    if len(heatmap_segments_utm) > 0:
        heatmap_gdf_utm = gpd.GeoDataFrame(
            {'valor': heatmap_values, 'geometry': heatmap_segments_utm},
            crs=gdf_utm.crs
        )
        heatmap_gdf_wgs = heatmap_gdf_utm.to_crs(epsg=4326)

    # Garante que a pasta de saída existe
    os.makedirs(pasta_saida_mapas, exist_ok=True)

    # Loop para gerar mapas com diferentes densidades
    passos = CONFIG.get('LISTA_PASSOS_VISUALIZACAO', [1])
    
    for passo in passos:
        print(f"Gerando mapa para passo de visualização: {passo} (1 a cada {passo})...")
        
        # Cria o objeto Mapa
        # Configura restrição de visão se solicitado
        kwargs_mapa = {
            'location': [centro_lat, centro_lon],
            'zoom_start': CONFIG['MAPA_ZOOM_INICIAL'],
            'max_zoom': CONFIG['MAPA_MAX_ZOOM'],
            'tiles': None,
            'attr_prefix': False # Remove o prefixo "Leaflet"
        }
        
        m = folium.Map(**kwargs_mapa)
        
        # Sempre ajusta o zoom para caber o projeto na tela (Responsividade)
        m.fit_bounds(bounds_projeto)
        
        # Prepara bounds para o TileLayer (Leaflet usa [[lat,lon], [lat,lon]])
        # O bounds_projeto já está nesse formato
        tile_bounds = bounds_projeto if CONFIG['MAPA_LIMITAR_VISAO_AO_PROJETO'] else None

        # Adiciona camada de Satélite (Google Satellite) - ÚNICA CAMADA BASE
        folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr = ' ', # Remove atribuição textual (vazio)
            name = 'Google Satélite',
            overlay = False,
            control = False, # Remove do controle de camadas
            max_zoom=CONFIG['MAPA_MAX_ZOOM'],
            max_native_zoom=20,
            bounds=tile_bounds
        ).add_to(m)

        # Cria FeatureGroups para controle de camadas
        fg_pontos = folium.FeatureGroup(name=CONFIG['LAYER_NAME_PONTOS'], show=CONFIG['LAYER_SHOW_PONTOS'])
        fg_linhas = folium.FeatureGroup(name=CONFIG['LAYER_NAME_LINHAS'], show=CONFIG['LAYER_SHOW_LINHAS'])
        fg_cotas_todas = folium.FeatureGroup(name=f"{CONFIG['LAYER_NAME_COTAS_OK']} (1/{passo})", show=CONFIG['LAYER_SHOW_COTAS_OK'])
        fg_cotas_alertas = folium.FeatureGroup(name=CONFIG['LAYER_NAME_COTAS_ALERTA'], show=CONFIG['LAYER_SHOW_COTAS_ALERTA'])
        fg_heatmap = folium.FeatureGroup(name=CONFIG['LAYER_NAME_HEATMAP'], show=CONFIG['LAYER_SHOW_HEATMAP'])

        # Adiciona os pontos ao FeatureGroup de pontos
        # OTIMIZAÇÃO: Só adiciona se o layer estiver visível ou se a configuração permitir gerar ocultos
        if CONFIG['GERAR_CAMADAS_OCULTAS'] or CONFIG['LAYER_SHOW_PONTOS']:
            for idx, row in gdf_mapa.iterrows():
                cor = CONFIG['PONTOS_COR_PADRAO']
                popup_txt = f"ID: {idx}"
                if 'AppliedRate' in row:
                    val = row['AppliedRate']
                    popup_txt += f"<br>Taxa: {val}"
                    if val > 0:
                        cor = CONFIG['PONTOS_COR_ATIVO']
                    else:
                        cor = CONFIG['PONTOS_COR_INATIVO']
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=CONFIG['PONTOS_RAIO'],
                    color=cor,
                    fill=True,
                    fill_color=cor,
                    popup=folium.Popup(popup_txt, max_width=200)
                ).add_to(fg_pontos)

        # Adiciona linhas geradas
        if (CONFIG['GERAR_CAMADAS_OCULTAS'] or CONFIG['LAYER_SHOW_LINHAS']) and len(linhas_geradas_wgs) > 0:
            for linha in linhas_geradas_wgs:
                coords = [(pt[1], pt[0]) for pt in linha.coords]
                folium.PolyLine(
                    locations=coords,
                    color=CONFIG['LINHAS_COR'],
                    weight=CONFIG['LINHAS_ESPESSURA'],
                    opacity=CONFIG['LINHAS_OPACIDADE']
                ).add_to(fg_linhas)

        # DESENHO DAS COTAS AUTOMÁTICAS
        # OTIMIZAÇÃO: Verifica se deve gerar cotas
        gerar_cotas_ok = CONFIG['GERAR_CAMADAS_OCULTAS'] or CONFIG['LAYER_SHOW_COTAS_OK']
        gerar_cotas_alerta = CONFIG['GERAR_CAMADAS_OCULTAS'] or CONFIG['LAYER_SHOW_COTAS_ALERTA']
        
        if (gerar_cotas_ok or gerar_cotas_alerta) and cotas_gdf_wgs is not None:
            for idx, row in cotas_gdf_wgs.iterrows():
                coords = [(pt[1], pt[0]) for pt in row.geometry.coords]
                dados_cota = row['valor']
                val = float(dados_cota['val'])
                is_alert = val < CONFIG['TOLERANCIA_MIN'] or val > CONFIG['TOLERANCIA_MAX']
                
                # Se for cota OK e não devemos gerar, pula
                if not is_alert and not gerar_cotas_ok:
                    continue
                # Se for alerta e não devemos gerar, pula
                if is_alert and not gerar_cotas_alerta:
                    continue

                # Filtro de Densidade Visual (Python-side)
                # Se não for alerta, aplica lógica de amostragem por zoom (Classes CSS)
                # is_alert cotas são sempre mostradas (exceto se layer desligada)
                
                # Classes de Amostragem:
                # cota-lvl-1: Mostra em Zoom Baixo (A cada 20 pontos)
                # cota-lvl-2: Mostra em Zoom Médio (A cada 5 pontos)
                # cota-lvl-3: Mostra em Zoom Alto (Todos os pontos)
                
                zoom_class = ""
                if is_alert:
                    zoom_class = "cota-alert"
                elif idx % 20 == 0:
                    zoom_class = "cota-lvl-1"
                elif idx % 5 == 0:
                    zoom_class = "cota-lvl-2"
                else:
                    zoom_class = "cota-lvl-3"
                    
                color = CONFIG['COTA_COR_ALERTA'] if is_alert else CONFIG['COTA_COR_OK']
                target_group = fg_cotas_alertas if is_alert else fg_cotas_todas
                
                mid_lat = (coords[0][0] + coords[1][0]) / 2
                mid_lon = (coords[0][1] + coords[1][1]) / 2
                
                # Adiciona a linha da cota
                # Linhas também recebem classe para sumir se necessário?
                # Por enquanto, apenas os labels (caixinhas) incomodam mais.
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=CONFIG['COTA_ESPESSURA'],
                    opacity=0.9,
                    popup=folium.Popup(f"Dist: {val:.2f} m", max_width=200)
                ).add_to(target_group)
                
                # Adiciona o rótulo
                text_color = 'white' if is_alert else 'black'
                bg_color = 'red' if is_alert else 'rgba(255,255,255,0.7)'
                border_color = color
                
                folium.map.Marker(
                    [mid_lat, mid_lon],
                    icon=folium.DivIcon(
                        icon_size=(150,36),
                        icon_anchor=(75,18),
                        class_name=f'cota-marker-container {zoom_class}',
                        html=f'<div style="font-size: 10pt; font-weight: bold; color: {text_color}; background: {bg_color}; border: 1px solid {border_color}; border-radius: 4px; text-align: center; width: auto; padding: 2px; display: inline-block;">{val:.2f}m</div>'
                    )
                ).add_to(target_group)

        # DESENHO DO MAPA DE CALOR
        if (CONFIG['GERAR_CAMADAS_OCULTAS'] or CONFIG['LAYER_SHOW_HEATMAP']) and heatmap_gdf_wgs is not None:
            for idx, row in heatmap_gdf_wgs.iterrows():
                val = row['valor']
                # Verifica se é alerta
                is_alert = val < CONFIG['TOLERANCIA_MIN'] or val > CONFIG['TOLERANCIA_MAX']
                
                # Define cor
                color = CONFIG['COTA_COR_ALERTA'] if is_alert else CONFIG['COTA_COR_OK']
                
                coords = [(pt[1], pt[0]) for pt in row.geometry.coords]
                
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=CONFIG['HEATMAP_ESPESSURA'],
                    opacity=CONFIG['HEATMAP_OPACIDADE']
                ).add_to(fg_heatmap)

        # Adiciona os FeatureGroups ao mapa
        # LÓGICA DE EXIBIÇÃO: Só adiciona ao mapa se tiver conteúdo E configuração permitir.
        # Se 'GERAR_CAMADAS_OCULTAS' for True, elas existem mas podem estar invisíveis (show=False).
        # Se for False, elas nem foram populadas.
        
        fg_heatmap.add_to(m)
        fg_cotas_alertas.add_to(m)
        
        # Camadas opcionais (só adiciona se foram geradas/populadas)
        if CONFIG['GERAR_CAMADAS_OCULTAS'] or CONFIG['LAYER_SHOW_PONTOS']:
            fg_pontos.add_to(m)
            
        if CONFIG['GERAR_CAMADAS_OCULTAS'] or CONFIG['LAYER_SHOW_LINHAS']:
            fg_linhas.add_to(m)
            
        if CONFIG['GERAR_CAMADAS_OCULTAS'] or CONFIG['LAYER_SHOW_COTAS_OK']:
            fg_cotas_todas.add_to(m)

        # Adiciona a legenda/estatísticas com CSS Responsivo e Controle de Zoom
        map_id = m.get_name()

        css_style = """
        <style>
            /* Remove a atribuição do Leaflet e Google */
            .leaflet-control-attribution {
                display: none !important;
            }
            
            /* Oculta cotas em zoom baixo (controlado via JS) */
            /* Por padrão, esconde niveis de detalhe (lvl 2 e 3) */
            /* Lvl 1 é sempre visível se não tiver classe global de hide */
            
            .cota-lvl-2, .cota-lvl-3 {
                display: none !important;
            }
            
            /* Quando tiver classe 'show-lvl-2' no mapa, mostra lvl 2 */
            .show-lvl-2 .cota-lvl-2 {
                display: inline-block !important;
            }
            
            /* Quando tiver classe 'show-lvl-3' no mapa, mostra lvl 3 */
            .show-lvl-3 .cota-lvl-3 {
                display: inline-block !important;
            }
            
            /* Classe global para esconder TUDO (zoom muito longe) */
            .hide-all-cotas .cota-marker-container {
                display: none !important;
            }

            .info-panel {
                position: fixed;
                bottom: 30px;
                left: 50%;
                transform: translateX(-50%);
                width: auto;
                min-width: 250px;
                max-width: 90vw;
                z-index: 9999;
                font-size: 14px;
                background-color: rgba(255, 255, 255, 0.95);
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                font-family: sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .info-panel table {
                width: 100%;
                font-size: 13px;
                margin-bottom: 5px;
            }
            .info-panel p {
                margin: 5px 0 0 0;
                font-size: 12px;
                text-align: center;
            }
            
            /* Estilo para o controle de layers quando movido para dentro do painel */
            .info-panel .leaflet-control-layers {
                box-shadow: none !important;
                border: none !important;
                background: none !important;
                margin-top: 5px;
                padding: 0;
                width: 100%;
            }
            
            .info-panel .leaflet-control-layers-list {
                margin-bottom: 0;
            }

            /* Ajustes para Celular (Telas pequenas) */
            @media (max-width: 600px) {
                .info-panel {
                    bottom: 10px;
                    width: 95vw;
                    left: 50%;
                    transform: translateX(-50%);
                    font-size: 11px;
                    padding: 8px;
                }
                .info-panel table {
                    font-size: 11px;
                }
                .info-panel p {
                    font-size: 10px;
                }
            }
        </style>
        """
        
        # Script para mover o LayerControl para dentro do InfoPanel
        js_move_layers = """
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                // Aguarda o carregamento do mapa e controles
                setTimeout(function() {
                    var layerControl = document.querySelector('.leaflet-control-layers');
                    var infoPanel = document.querySelector('.info-panel');
                    
                    if (layerControl && infoPanel) {
                        // Move o controle para dentro do painel
                        infoPanel.appendChild(layerControl);
                    }
                }, 800);
            });
        </script>
        """
        
        # Script para controlar visibilidade das cotas baseado no Zoom
        js_zoom_logic = f"""
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                // Acessa o objeto mapa pelo ID gerado pelo Folium
                var map = {map_id};
                
                // Define o zoom mínimo para exibir as cotas
                // Lógica de Amostragem por Nível de Zoom
                
                // Aguarda um pouco para o fitBounds ocorrer
                setTimeout(function() {{
                    var initialZoom = map.getZoom();
                    
                    // Definição de Limiares (Thresholds) relativos ao zoom inicial
                    // Zoom Baixo (Longe): Mostra apenas lvl 1 (1/20)
                    // Zoom Médio: Mostra lvl 1 + lvl 2 (1/5)
                    // Zoom Alto (Perto): Mostra tudo (1/1)
                    
                    // Exemplo: Se Initial = 18.
                    // < 15: Esconde tudo (Opcional, ou mostra só lvl 1)
                    // 15 - 16: Lvl 1
                    // 17: Lvl 1 + 2
                    // >= 18: Tudo
                    
                    var zoomThresholdLow = initialZoom - 4; // Abaixo disso, esconde tudo ou mostra muito pouco
                    var zoomThresholdMid = initialZoom - 2; // Começa a mostrar médio
                    var zoomThresholdHigh = initialZoom;    // Mostra tudo
                    
                    console.log("Zoom Logic Init: Initial=" + initialZoom);

                    function updateCotaVisibility() {{
                        var currentZoom = map.getZoom();
                        var mapContainer = map.getContainer();
                        var classList = mapContainer.classList;
                        
                        // Limpa estados
                        classList.remove('hide-all-cotas');
                        classList.remove('show-lvl-2');
                        classList.remove('show-lvl-3');
                        
                        if (currentZoom < zoomThresholdLow) {{
                            // Zoom muito longe: Esconde tudo para limpar a visão
                            classList.add('hide-all-cotas');
                        }} else if (currentZoom < zoomThresholdMid) {{
                            // Zoom baixo: Mostra apenas Lvl 1 (padrão CSS, sem classes extras)
                            // Nenhuma ação necessária, pois lvl 2 e 3 estão hidden por CSS padrão
                        }} else if (currentZoom < zoomThresholdHigh) {{
                            // Zoom médio: Mostra Lvl 1 + Lvl 2
                            classList.add('show-lvl-2');
                        }} else {{
                            // Zoom alto: Mostra Tudo
                            classList.add('show-lvl-2');
                            classList.add('show-lvl-3');
                        }}
                    }}
                    
                    map.on('zoomend', updateCotaVisibility);
                    
                    // Executa verificação inicial
                    updateCotaVisibility();
                }}, 1000);
            }});
        </script>
        """
        
        m.get_root().html.add_child(folium.Element(css_style))
        m.get_root().html.add_child(folium.Element(js_move_layers))
        m.get_root().html.add_child(folium.Element(js_zoom_logic))

        legend_html = f'''
        <div class="info-panel">
            {stats_html}
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Controles finais
        # Adiciona LayerControl expandido. A posição inicial não importa muito pois o JS vai mover,
        # mas 'bottomright' evita sobreposição inicial.
        folium.LayerControl(collapsed=False, position='bottomright').add_to(m)
        
        Fullscreen().add_to(m)

        # Calcula data de ontem
        ontem = datetime.now() - timedelta(days=1)
        data_str = ontem.strftime('%d-%m-%Y') # Usando hifens para ser válido no sistema de arquivos
        
        # Nome base com a data
        nome_base = f"Mapa Espaçamento Plantio - {data_str}"

        # Salva o arquivo específico (com sufixo de densidade para diferenciação técnica)
        nome_arquivo = f'{nome_base} - densidade_{passo}.html'
        output_file = os.path.join(pasta_saida_mapas, nome_arquivo)
        m.save(output_file)
        print(f"Mapa salvo com sucesso em: {output_file}")
        
        # Se for o passo 1 (padrão), salva com o nome principal solicitado pelo usuário
        if passo == 1:
             output_file_default = os.path.join(pasta_saida_mapas, f'{nome_base}.html')
             m.save(output_file_default)
             print(f"Mapa principal salvo em: {output_file_default}")

    print("Todos os mapas foram gerados com sucesso.")

if __name__ == "__main__":
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Busca o arquivo zip
    zip_encontrado = localizar_zip_na_pasta(diretorio_atual)
    
    if zip_encontrado:
        print(f"Arquivo ZIP encontrado: {zip_encontrado}")
        pasta_extraida = os.path.join(diretorio_atual, "dados_extraidos")
        extrair_zip(zip_encontrado, pasta_extraida)
        pasta_dados = pasta_extraida
    else:
        print("Nenhum ZIP encontrado. Usando a pasta 'doc' atual.")
        pasta_dados = os.path.join(diretorio_atual, "doc")
    
    # 2. Procura o shapefile dentro da pasta de dados
    arquivos_shp = glob.glob(os.path.join(pasta_dados, "**", "*.shp"), recursive=True)
    
    if not arquivos_shp:
        print("Nenhum arquivo .shp encontrado.")
    else:
        arquivo_alvo = arquivos_shp[0]
        pasta_mapas = os.path.join(diretorio_atual, "mapas")
        analisar_espacamento_e_gerar_mapa(arquivo_alvo, pasta_mapas)
