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
    'COTAS_MOSTRAR_TODAS': False, # Se True, mostra até as que estão OK (pode poluir)
    'COTAS_MOSTRAR_ALERTAS': True, # Mostra as que estão fora da tolerância
    'COTA_COR_OK': '#00FF00', # Verde (dentro do padrão)
    'COTA_COR_ALERTA': '#FF00FF', # Magenta (fora do padrão - ALERTA)
    'COTA_ESPESSURA': 2,
    
    # Parâmetros de Tolerância (O que é aceitável?)
    'TOLERANCIA_MIN': 2.5, # Abaixo disso é alerta
    'TOLERANCIA_MAX': 3.5, # Acima disso é alerta
    
    # Parâmetros de Geração de Linha
    'QUEBRA_TEMPO_SEC': 60,
    'QUEBRA_ANGULO_GRAUS': 60,
    
    # Parâmetros de Cálculo de Espaçamento
    # NOTA: Agora o Python gera um SUPERSET (alta densidade).
    # O filtro real de amostragem e offset será feito no JS.
    'OFFSET_CABECEIRA_METROS': 0.0, # Python gera tudo, JS filtra
    'AMOSTRAGEM_DIST_METROS': 1.0, # Alta densidade para o JS ter liberdade
    'FILTRO_DIST_MIN': 0.1, 
    'FILTRO_DIST_MAX': 100.0, 
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
                
                # Cria pontos de amostragem a cada X metros (ALTA DENSIDADE para o Frontend filtrar)
                # Exportamos metadados de posição para o JS poder aplicar o Offset dinamicamente
                
                # Gerar pontos a cada 1 metro
                distancias_amostragem = np.arange(0, length, CONFIG['AMOSTRAGEM_DIST_METROS'])
                if len(distancias_amostragem) == 0:
                     distancias_amostragem = [length / 2]
                
                pontos_amostra = [geom.interpolate(d) for d in distancias_amostragem]
                
                for i, pt in enumerate(pontos_amostra):
                    dist_from_start = distancias_amostragem[i]
                    
                    # Busca linhas candidatas num raio de busca (ex: 50m)
                    candidatos_idx = list(sindex.intersection(pt.buffer(CONFIG['FILTRO_DIST_MAX']).bounds))
                    candidatos = linhas_gdf.iloc[candidatos_idx]
                    
                    min_dist = float('inf')
                    found_neighbor = False
                    ponto_vizinho_mais_proximo = None
                    
                    for cand_idx, cand_row in candidatos.iterrows():
                        if cand_idx == idx:
                            continue # Pula a própria linha
                        
                        # Para achar o ponto exato na linha vizinha, usamos nearest_points da shapely ops
                        from shapely.ops import nearest_points
                        p1, p2 = nearest_points(pt, cand_row.geometry)
                        
                        dist = p1.distance(p2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            found_neighbor = True
                            ponto_vizinho_mais_proximo = p2
                    
                    if found_neighbor:
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
    # 4. GERAÇÃO DO MAPA INTERATIVO (LEAFLET)
    # ---------------------------------------------------------
    print("\n--- Gerando Mapa Interativo ---")
    
    # Usa o GeoDataFrame original (WGS84) para o mapa, pois Leaflet usa Lat/Lon
    # Mas vamos simplificar a geometria se for muito pesada
    gdf_mapa = gdf.copy()
    
    # Se tiver muitos pontos (> 5000), faz uma amostragem para o mapa não travar
    if len(gdf_mapa) > 5000:
        print(f"Reduzindo pontos para visualização (de {len(gdf_mapa)} para 5000)...")
        gdf_mapa = gdf_mapa.sample(5000)
    
    # Pega o centro do mapa
    centro_lat = gdf_mapa.geometry.y.mean()
    centro_lon = gdf_mapa.geometry.x.mean()
    
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
    
    # Adiciona camada de Satélite (Google Satellite) com maxNativeZoom
    # maxNativeZoom diz ao Leaflet: "Eu tenho imagens até zoom 20, mas se o usuário der zoom 22, apenas estique a imagem do 20"
    folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satélite',
        overlay = False,
        control = True,
        max_zoom=CONFIG['MAPA_MAX_ZOOM'],
        max_native_zoom=20 
    ).add_to(m)

    # Cria FeatureGroups para controle de camadas (Pontos e Linhas)
    fg_pontos = folium.FeatureGroup(name="Pontos (Original)", show=CONFIG['PONTOS_MOSTRAR'])
    fg_linhas = folium.FeatureGroup(name="Linhas (Calculadas)", show=CONFIG['LINHAS_MOSTRAR'])
    # Removido grupos estáticos de cotas pois agora serão dinâmicos via JS
    # fg_cotas_todas = ...
    # fg_cotas_alertas = ...

    # Adiciona os pontos ao FeatureGroup de pontos
    # Usamos CircleMarker para performance e visual
    for idx, row in gdf_mapa.iterrows():
        # Cor baseada no AppliedRate se existir, senão padrão
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

    # Adiciona linhas geradas ao FeatureGroup de linhas
    if len(linhas_geradas_wgs) > 0:
        for linha in linhas_geradas_wgs:
            coords = [(pt[1], pt[0]) for pt in linha.coords]
            folium.PolyLine(
                locations=coords,
                color=CONFIG['LINHAS_COR'],
                weight=CONFIG['LINHAS_ESPESSURA'],
                opacity=CONFIG['LINHAS_OPACIDADE']
            ).add_to(fg_linhas)

    # -------------------------------------------------------------------------
    # DESENHO DAS COTAS AUTOMÁTICAS (VIA JS)
    # -------------------------------------------------------------------------
    # Vamos exportar os dados das cotas para uma variável JS e renderizar no cliente.
    # Isso permite alterar cores e filtros sem recarregar o Python.
    
    cotas_json_data = []
    
    if len(cotas_geometry) > 0:
        cotas_gdf_utm = gpd.GeoDataFrame(
            {'valor': cotas_valores, 'geometry': cotas_geometry}, 
            crs=gdf_utm.crs
        )
        # Converte para WGS84 (Lat/Lon)
        cotas_gdf_wgs = cotas_gdf_utm.to_crs(epsg=4326)
        
        for idx, row in cotas_gdf_wgs.iterrows():
            coords = [[pt[1], pt[0]] for pt in row.geometry.coords] # [[lat, lon], [lat, lon]]
            cotas_json_data.append({
                "coords": coords,
                "val": round(row['valor'], 2)
            })

    # Adiciona os FeatureGroups estáticos ao mapa
    fg_pontos.add_to(m)
    fg_linhas.add_to(m)
    
    # -------------------------------------------------------------------------
    # PAINEL DE CONTROLE JS E RENDERIZAÇÃO
    # -------------------------------------------------------------------------
    
    import json
    cotas_data_str = json.dumps(cotas_json_data)
    
    control_panel_html = f"""
    <div id="control-panel" style="
        position: fixed; 
        top: 10px; left: 60px; 
        width: 250px; 
        background: white; 
        padding: 10px; 
        border: 2px solid rgba(0,0,0,0.2); 
        z-index: 1000;
        border-radius: 5px;
        font-family: sans-serif;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    ">
        <h4 style="margin-top:0;">🔧 Controles</h4>
        
        <label><b>Tolerância Mín (m):</b> <span id="lbl_min">{CONFIG['TOLERANCIA_MIN']}</span></label>
        <input type="range" id="rng_min" min="0" max="5" step="0.1" value="{CONFIG['TOLERANCIA_MIN']}" style="width:100%">
        
        <label><b>Tolerância Máx (m):</b> <span id="lbl_max">{CONFIG['TOLERANCIA_MAX']}</span></label>
        <input type="range" id="rng_max" min="0" max="10" step="0.1" value="{CONFIG['TOLERANCIA_MAX']}" style="width:100%">
        
        <hr>
        <label><input type="checkbox" id="chk_labels" checked> Mostrar Rótulos (Distância)</label>
        <br>
        <label><input type="checkbox" id="chk_cotas" checked> Mostrar Linhas de Cota</label>
        
        <hr>
        <small>Total Cotas: {len(cotas_json_data)}</small>
    </div>

    <script>
    // Dados injetados pelo Python
    var cotasData = {cotas_data_str};
    
    var cotasLayerGroup = L.layerGroup().addTo(map);
    var labelsLayerGroup = L.layerGroup().addTo(map);
    
    var minTol = {CONFIG['TOLERANCIA_MIN']};
    var maxTol = {CONFIG['TOLERANCIA_MAX']};
    var showLabels = true;
    var showCotas = true;
    
    function updateMap() {{
        cotasLayerGroup.clearLayers();
        labelsLayerGroup.clearLayers();
        
        if (!showCotas) return;
        
        cotasData.forEach(function(cota) {{
            var val = cota.val;
            var color = '{CONFIG['COTA_COR_OK']}'; // Verde
            var isAlert = false;
            
            if (val < minTol || val > maxTol) {{
                color = '{CONFIG['COTA_COR_ALERTA']}'; // Magenta
                isAlert = true;
            }}
            
            // Desenha linha
            L.polyline(cota.coords, {{
                color: color,
                weight: isAlert ? 3 : 1, // Alerta mais grosso
                opacity: 0.8
            }}).addTo(cotasLayerGroup).bindPopup("Dist: " + val + " m");
            
            // Desenha Rótulo (apenas se checkbox ativo)
            // Para não poluir, mostramos rótulos apenas para alertas ou se o zoom for alto?
            // O usuário pediu "já deixar visivel". Vamos colocar em todos por enquanto.
            if (showLabels) {{
                var p1 = L.latLng(cota.coords[0]);
                var p2 = L.latLng(cota.coords[1]);
                var center = L.latLng((p1.lat + p2.lat)/2, (p1.lng + p2.lng)/2);
                
                var myIcon = L.divIcon({{
                    className: 'cota-label',
                    html: '<div style="color:' + color + '; font-weight:bold; font-size:10px; text-shadow: 1px 1px 1px black;">' + val + '</div>',
                    iconSize: [30, 10],
                    iconAnchor: [15, 5]
                }});
                
                L.marker(center, {{icon: myIcon}}).addTo(labelsLayerGroup);
            }}
        }});
    }}
    
    // Event Listeners
    document.getElementById('rng_min').addEventListener('input', function(e) {{
        minTol = parseFloat(e.target.value);
        document.getElementById('lbl_min').innerText = minTol;
        updateMap();
    }});
    
    document.getElementById('rng_max').addEventListener('input', function(e) {{
        maxTol = parseFloat(e.target.value);
        document.getElementById('lbl_max').innerText = maxTol;
        updateMap();
    }});
    
    document.getElementById('chk_labels').addEventListener('change', function(e) {{
        showLabels = e.target.checked;
        updateMap();
    }});
    
    document.getElementById('chk_cotas').addEventListener('change', function(e) {{
        showCotas = e.target.checked;
        updateMap();
    }});
    
    // Inicializa
    updateMap();
    
    // Ajusta z-index do controle para ficar acima do mapa
    var mapContainer = document.querySelector('.leaflet-container');
    
    </script>
    """
    m.get_root().html.add_child(folium.Element(control_panel_html))

    # Adiciona a tabela de estatísticas como um elemento flutuante no mapa
    # Usamos HTML/CSS para posicionar
    legend_html = f'''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 300px; height: auto; 
     z-index:9999; font-size:14px;
     background-color: white;
     border: 2px solid grey;
     padding: 10px;
     opacity: 0.9;">
     {stats_html}
     <p><small><i>Gerado automaticamente pelo script Python</i></small></p>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # -------------------------------------------------------------------------
    # SCRIPT CUSTOMIZADO: RÉGUA RÁPIDA (2 Cliques)
    # -------------------------------------------------------------------------
    # Injeta JavaScript para criar uma ferramenta de medição simples
    custom_js = """
    <script>
    var measuring = false;
    var measurePoints = [];
    var measureLine = null;
    var measureLabel = null;
    
    // Cria um botão personalizado no mapa
    var btn = L.DomUtil.create('button', 'leaflet-bar leaflet-control leaflet-control-custom');
    btn.innerHTML = '📏 Régua Rápida';
    btn.style.backgroundColor = 'white';
    btn.style.width = '120px';
    btn.style.height = '30px';
    btn.style.cursor = 'pointer';
    btn.onclick = function(){
        measuring = !measuring;
        measurePoints = [];
        if(measuring){
            btn.innerHTML = '❌ Cancelar';
            btn.style.backgroundColor = '#ffcccc';
            document.getElementById('map').style.cursor = 'crosshair';
        } else {
            btn.innerHTML = '📏 Régua Rápida';
            btn.style.backgroundColor = 'white';
            document.getElementById('map').style.cursor = '';
            if(measureLine) { map.removeLayer(measureLine); }
            if(measureLabel) { map.removeLayer(measureLabel); }
        }
    }
    
    var container = document.getElementsByClassName('leaflet-top leaflet-right')[0];
    container.appendChild(btn);
    
    // Evento de clique no mapa
    var map = document.querySelector('.leaflet-container')._leaflet_map;
    
    map.on('click', function(e) {
        if (!measuring) return;
        
        measurePoints.push(e.latlng);
        
        // Marcador temporário no ponto
        L.circleMarker(e.latlng, {radius: 3, color: 'black'}).addTo(map).bindPopup("Pt " + measurePoints.length).openPopup();
        
        if (measurePoints.length == 2) {
            var p1 = measurePoints[0];
            var p2 = measurePoints[1];
            
            // Calcula distância (metros)
            var dist = p1.distanceTo(p2).toFixed(2);
            
            // Desenha linha
            measureLine = L.polyline([p1, p2], {color: 'yellow', dashArray: '10, 10', weight: 3}).addTo(map);
            
            // Ponto médio para o rótulo
            var midLat = (p1.lat + p2.lat) / 2;
            var midLng = (p1.lng + p2.lng) / 2;
            
            measureLabel = L.marker([midLat, midLng], {
                icon: L.divIcon({
                    className: 'measure-label',
                    html: '<div style="background:white; border:1px solid black; padding:2px;">' + dist + ' m</div>'
                })
            }).addTo(map);
            
            // Reseta para próxima medição (ou encerra, dependendo do gosto)
            // Aqui vamos resetar os pontos mas manter a linha antiga até clicar de novo? 
            // Melhor resetar tudo no terceiro clique ou deixar acumular?
            // O usuário pediu "2 cliques". Vamos encerrar o modo de medição ou limpar array.
            
            measurePoints = []; // Limpa para nova medição imediata
            // measuring = false; // Descomente se quiser sair do modo régua após 1 medição
        }
    });
    </script>
    """
    m.get_root().html.add_child(folium.Element(custom_js))

    # Adiciona controles extras
    folium.LayerControl().add_to(m) # Controle de camadas
    MeasureControl(primary_length_unit='meters').add_to(m) # Ferramenta de régua padrão (backup)
    Fullscreen().add_to(m) # Botão de tela cheia

    # Salva o arquivo
    os.makedirs(pasta_saida_mapas, exist_ok=True)
    output_file = os.path.join(pasta_saida_mapas, 'mapa_projeto.html')
    m.save(output_file)
    print(f"Mapa salvo com sucesso em: {output_file}")
    print("Abra este arquivo no seu navegador para ver o mapa e a tabela.")

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
