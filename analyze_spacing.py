import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
import math
import os
import glob
import zipfile
import folium
from folium.plugins import MeasureControl, Fullscreen

# ==============================================================================
# CONFIGURAÇÕES GERAIS - Ajuste aqui os parâmetros de análise e visualização
# ==============================================================================
CONFIG = {
    # Visualização do Mapa
    'MAPA_ZOOM_INICIAL': 18,
    'MAPA_MAX_ZOOM': 24, # Permite zoom muito alto
    
    # Estilo dos Pontos
    'PONTOS_MOSTRAR': True,
    'PONTOS_RAIO': 1,
    'PONTOS_COR_PADRAO': '#FF4500', # OrangeRed
    'PONTOS_COR_ATIVO': '#00FF00', # Lime
    'PONTOS_COR_INATIVO': '#FF0000', # Red
    
    # Estilo das Linhas Geradas (Passadas)
    'LINHAS_MOSTRAR': True,
    'LINHAS_COR': '#00FFFF', # Cyan
    'LINHAS_ESPESSURA': 3,
    'LINHAS_OPACIDADE': 0.6,
    
    # Estilo das Cotas Automáticas (Medições)
    'COTAS_MOSTRAR_TODAS': True,
    'COTAS_MOSTRAR_ALERTAS': True, 
    'LISTA_PASSOS_VISUALIZACAO': [10], # Gera mapas para cada passo: 1=Todas, 2=50%, 5=20%, etc.
    'COTA_COR_OK': '#00FF00', # Verde (dentro do padrão)
    'COTA_COR_ALERTA': '#FF0000', # Vermelho (fora do padrão - ALERTA)
    'COTA_ESPESSURA': 2,
    
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
    stats_html = "<h3>Relatório de Espaçamento</h3>"
    
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
            
            # Adiciona ao HTML
            stats_html += f"""
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr><td><b>Média (Robusta)</b></td><td>{media:.2f} m</td></tr>
                <tr><td><b>Mediana (Robusta)</b></td><td>{mediana:.2f} m</td></tr>
                <tr><td><b>Mínimo</b></td><td>{minimo:.2f} m</td></tr>
                <tr><td><b>Máximo</b></td><td>{maximo:.2f} m</td></tr>
                <tr><td><b>Amostras Totais</b></td><td>{len(distancias_validas)}</td></tr>
                <tr><td><b>Amostras Úteis</b></td><td>{len(distancias_robustas)}</td></tr>
            </table>
            <p><small>*Valores robustos ignoram outliers (IQR filter).</small></p>
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
    
    # Prepara cotas em WGS84 uma única vez
    cotas_gdf_wgs = None
    if len(cotas_geometry) > 0:
        cotas_gdf_utm = gpd.GeoDataFrame(
            {'valor': cotas_valores, 'geometry': cotas_geometry}, 
            crs=gdf_utm.crs
        )
        cotas_gdf_wgs = cotas_gdf_utm.to_crs(epsg=4326)

    # Garante que a pasta de saída existe
    os.makedirs(pasta_saida_mapas, exist_ok=True)

    # Loop para gerar mapas com diferentes densidades
    passos = CONFIG.get('LISTA_PASSOS_VISUALIZACAO', [1])
    
    for passo in passos:
        print(f"Gerando mapa para passo de visualização: {passo} (1 a cada {passo})...")
        
        # Cria o objeto Mapa
        m = folium.Map(
            location=[centro_lat, centro_lon], 
            zoom_start=CONFIG['MAPA_ZOOM_INICIAL'], 
            max_zoom=CONFIG['MAPA_MAX_ZOOM'],
            tiles=None # Vamos adicionar tiles manualmente para controle total
        )
        
        # Adiciona camada Base (OpenStreetMap)
        folium.TileLayer(
            'OpenStreetMap',
            name='Mapa de Rua',
            max_zoom=CONFIG['MAPA_MAX_ZOOM'],
            control=True
        ).add_to(m)
        
        # Adiciona camada de Satélite (Google Satellite)
        folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Satélite',
            overlay = False,
            control = True,
            max_zoom=CONFIG['MAPA_MAX_ZOOM'],
            max_native_zoom=20 
        ).add_to(m)

        # Cria FeatureGroups para controle de camadas
        fg_pontos = folium.FeatureGroup(name="Pontos (Original)", show=CONFIG['PONTOS_MOSTRAR'])
        fg_linhas = folium.FeatureGroup(name="Linhas (Calculadas)", show=CONFIG['LINHAS_MOSTRAR'])
        fg_cotas_todas = folium.FeatureGroup(name=f"Cotas (OK - 1/{passo})", show=CONFIG['COTAS_MOSTRAR_TODAS'])
        fg_cotas_alertas = folium.FeatureGroup(name="Cotas (Alerta)", show=CONFIG['COTAS_MOSTRAR_ALERTAS'])

        # Adiciona os pontos ao FeatureGroup de pontos
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
        if len(linhas_geradas_wgs) > 0:
            for linha in linhas_geradas_wgs:
                coords = [(pt[1], pt[0]) for pt in linha.coords]
                folium.PolyLine(
                    locations=coords,
                    color=CONFIG['LINHAS_COR'],
                    weight=CONFIG['LINHAS_ESPESSURA'],
                    opacity=CONFIG['LINHAS_OPACIDADE']
                ).add_to(fg_linhas)

        # DESENHO DAS COTAS AUTOMÁTICAS
        if cotas_gdf_wgs is not None:
            for idx, row in cotas_gdf_wgs.iterrows():
                coords = [(pt[1], pt[0]) for pt in row.geometry.coords]
                dados_cota = row['valor']
                val = float(dados_cota['val'])
                is_alert = val < CONFIG['TOLERANCIA_MIN'] or val > CONFIG['TOLERANCIA_MAX']
                
                # Filtro de Densidade Visual (Python-side)
                # Se não for alerta, aplica o passo de visualização
                if not is_alert and (idx % passo != 0):
                    continue
                    
                color = CONFIG['COTA_COR_ALERTA'] if is_alert else CONFIG['COTA_COR_OK']
                target_group = fg_cotas_alertas if is_alert else fg_cotas_todas
                
                mid_lat = (coords[0][0] + coords[1][0]) / 2
                mid_lon = (coords[0][1] + coords[1][1]) / 2
                
                # Adiciona a linha da cota
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
                        html=f'<div style="font-size: 10pt; font-weight: bold; color: {text_color}; background: {bg_color}; border: 1px solid {border_color}; border-radius: 4px; text-align: center; width: auto; padding: 2px; display: inline-block;">{val:.2f}m</div>'
                    )
                ).add_to(target_group)

        # Adiciona os FeatureGroups ao mapa
        fg_pontos.add_to(m)
        fg_linhas.add_to(m)
        fg_cotas_todas.add_to(m)
        fg_cotas_alertas.add_to(m)

        # Adiciona a tabela de estatísticas
        legend_html = f'''
         <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 320px; height: auto; 
         z-index:9999; font-size:14px;
         background-color: white;
         border: 2px solid grey;
         padding: 10px;
         opacity: 0.95;
         box-shadow: 0 0 10px rgba(0,0,0,0.2);">
         
         {stats_html}
         <p><small><i>Densidade Visual: 1 a cada {passo} cotas (OK).</i></small></p>
         </div>
         '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Controles finais
        folium.LayerControl().add_to(m)
        Fullscreen().add_to(m)

        # Salva o arquivo específico
        nome_arquivo = f'mapa_projeto_densidade_{passo}.html'
        output_file = os.path.join(pasta_saida_mapas, nome_arquivo)
        m.save(output_file)
        print(f"Mapa salvo com sucesso em: {output_file}")
        
        # Se for o passo 1 (padrão), salva também como mapa_projeto.html para facilitar
        if passo == 1:
             output_file_default = os.path.join(pasta_saida_mapas, 'mapa_projeto.html')
             m.save(output_file_default)

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
