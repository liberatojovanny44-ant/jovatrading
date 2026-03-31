import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════
#   JOVATRADING PRO — 4 Pares | 12 Indicadores | SL/TP | Backtesting
# ═══════════════════════════════════════════════════════════

API_KEY    = 'ekg9CUK8G6N2P6gxxh7xfN10QWKkb8nuRqd329b4c9tMNRgSXSsd3ZGUhCIiaDyM'.strip().replace('\n', '').replace(' ', '')
SECRET_KEY = 'zYcWvR7e4tlsa2a3QIntNxW9mYXgwxe5v1LavPRY3WhpGy2SuIPNy4TgRZq1fA1A'.strip().replace('\n', '').replace(' ', '')

PARES = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']

RIESGO_PCT      = 0.02   # 2% del balance por operación
RATIO_RIESGO    = 3.0    # Take Profit = 3x el riesgo (ratio 1:3)
MIN_CONFIANZA   = 4      # Mínimo de indicadores a favor para operar
CAPITAL_INICIAL = 10000  # Para backtesting

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    }
})
exchange.set_sandbox_mode(True)

archivo_registro = 'historial_jovatrading_pro.csv'

# ═══════════════════════════════════════════════════════════
#   INDICADORES TÉCNICOS (12)
# ═══════════════════════════════════════════════════════════

def calcular_indicadores(df):
    c = df['close'].copy()
    h = df['high'].copy()
    l = df['low'].copy()
    v = df['vol'].copy()

    # 1. EMA 9
    df['ema9']  = c.ewm(span=9,   adjust=False).mean()
    # 2. EMA 21
    df['ema21'] = c.ewm(span=21,  adjust=False).mean()
    # 3. EMA 200
    df['ema200']= c.ewm(span=200, adjust=False).mean()

    # 4. RSI (14)
    delta = c.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    # 5. MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # 6. Bandas de Bollinger (20, 2)
    sma20          = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_mid']   = sma20

    # 7. Estocástico (14)
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # 8. ATR (14) — para calcular SL dinámico
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # 9. ADX (14) — fuerza de tendencia
    up   = h.diff()
    down = -l.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)
    atr14    = tr.rolling(14).mean()
    df['adx_plus']  = 100 * (plus_dm.rolling(14).mean()  / atr14)
    df['adx_minus'] = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * (df['adx_plus'] - df['adx_minus']).abs() / (df['adx_plus'] + df['adx_minus'] + 1e-9)
    df['adx'] = dx.rolling(14).mean()

    # 10. Volumen relativo (vs promedio 20 velas)
    df['vol_avg'] = v.rolling(20).mean()
    df['vol_rel'] = v / df['vol_avg']

    # 11. Soporte y Resistencia (últimas 20 velas)
    df['soporte']     = l.rolling(20).min()
    df['resistencia'] = h.rolling(20).max()

    # 12. Momentum (ROC 10)
    df['roc'] = ((c - c.shift(10)) / c.shift(10)) * 100

    return df

# ═══════════════════════════════════════════════════════════
#   MOTOR DE SEÑALES
# ═══════════════════════════════════════════════════════════

def analizar_señal(df, symbol):
    row = df.iloc[-1]
    prev = df.iloc[-2]

    puntos_buy  = 0
    puntos_sell = 0
    razones_buy  = []
    razones_sell = []

    precio = row['close']

    # 1. EMA Cruce dorado/muerte
    if prev['ema9'] <= prev['ema21'] and row['ema9'] > row['ema21']:
        puntos_buy += 1; razones_buy.append('Cruce dorado EMA9/21')
    if prev['ema9'] >= prev['ema21'] and row['ema9'] < row['ema21']:
        puntos_sell += 1; razones_sell.append('Cruce muerte EMA9/21')

    # 2. Precio vs EMA 200
    if precio > row['ema200']:
        puntos_buy += 1; razones_buy.append('Precio sobre EMA200')
    else:
        puntos_sell += 1; razones_sell.append('Precio bajo EMA200')

    # 3. RSI
    if row['rsi'] < 35:
        puntos_buy += 1; razones_buy.append(f'RSI sobrevendido ({row["rsi"]:.1f})')
    if row['rsi'] > 65:
        puntos_sell += 1; razones_sell.append(f'RSI sobrecomprado ({row["rsi"]:.1f})')

    # 4. MACD
    if prev['macd_hist'] < 0 and row['macd_hist'] > 0:
        puntos_buy += 1; razones_buy.append('MACD cruce alcista')
    if prev['macd_hist'] > 0 and row['macd_hist'] < 0:
        puntos_sell += 1; razones_sell.append('MACD cruce bajista')
    if row['macd'] > row['macd_signal']:
        puntos_buy += 0.5
    else:
        puntos_sell += 0.5

    # 5. Bollinger Bands
    if precio <= row['bb_lower']:
        puntos_buy += 1; razones_buy.append('Precio en BB inferior')
    if precio >= row['bb_upper']:
        puntos_sell += 1; razones_sell.append('Precio en BB superior')

    # 6. Estocástico
    if row['stoch_k'] < 20 and row['stoch_k'] > row['stoch_d']:
        puntos_buy += 1; razones_buy.append(f'Estocástico sobrevendido ({row["stoch_k"]:.1f})')
    if row['stoch_k'] > 80 and row['stoch_k'] < row['stoch_d']:
        puntos_sell += 1; razones_sell.append(f'Estocástico sobrecomprado ({row["stoch_k"]:.1f})')

    # 7. ADX — solo operar si hay tendencia fuerte
    tendencia_fuerte = row['adx'] > 25

    # 8. Volumen alto confirma señal
    if row['vol_rel'] > 1.5:
        if puntos_buy > puntos_sell:
            puntos_buy += 1; razones_buy.append('Volumen alto confirma')
        else:
            puntos_sell += 1; razones_sell.append('Volumen alto confirma')

    # 9. Soporte/Resistencia
    margen = (row['resistencia'] - row['soporte']) * 0.05
    if abs(precio - row['soporte']) < margen:
        puntos_buy += 1; razones_buy.append('Precio en soporte')
    if abs(precio - row['resistencia']) < margen:
        puntos_sell += 1; razones_sell.append('Precio en resistencia')

    # 10. Momentum (ROC)
    if row['roc'] > 0 and prev['roc'] < 0:
        puntos_buy += 1; razones_buy.append('Momentum positivo')
    if row['roc'] < 0 and prev['roc'] > 0:
        puntos_sell += 1; razones_sell.append('Momentum negativo')

    # Calcular SL y TP usando ATR
    atr = row['atr']
    sl_dist = atr * 1.5
    tp_dist = sl_dist * RATIO_RIESGO

    if puntos_buy >= MIN_CONFIANZA and puntos_buy > puntos_sell and tendencia_fuerte:
        sl = precio - sl_dist
        tp = precio + tp_dist
        return {
            'accion': 'COMPRA',
            'confianza': int(puntos_buy),
            'razon': ' | '.join(razones_buy),
            'sl': sl,
            'tp': tp,
            'precio': precio,
            'atr': atr
        }
    elif puntos_sell >= MIN_CONFIANZA and puntos_sell > puntos_buy and tendencia_fuerte:
        return {
            'accion': 'VENTA_SEÑAL',
            'confianza': int(puntos_sell),
            'razon': ' | '.join(razones_sell),
            'sl': None,
            'tp': None,
            'precio': precio,
            'atr': atr
        }
    else:
        todas_razones = razones_buy + razones_sell
        return {
            'accion': 'OBSERVANDO',
            'confianza': 0,
            'razon': ' | '.join(todas_razones) if todas_razones else 'Sin señal clara',
            'sl': None,
            'tp': None,
            'precio': precio,
            'atr': atr
        }

# ═══════════════════════════════════════════════════════════
#   BACKTESTING 6 MESES
# ═══════════════════════════════════════════════════════════

def ejecutar_backtesting():
    print('\n' + '═'*55)
    print('  BACKTESTING — 6 MESES DE DATOS HISTÓRICOS')
    print('═'*55)

    resultados_totales = []

    for symbol in PARES:
        print(f'\n📊 Backtesting {symbol}...')

        try:
            # Descargar ~6 meses de velas de 1 hora (4320 velas)
            velas = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=4320)
            df = pd.DataFrame(velas, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            df = calcular_indicadores(df)
            df.dropna(inplace=True)

            capital    = CAPITAL_INICIAL
            operaciones = []
            en_posicion = False
            entrada_precio = 0
            sl_precio = 0
            tp_precio = 0
            cantidad_btc = 0

            for i in range(len(df) - 1):
                ventana = df.iloc[:i+1]
                if len(ventana) < 50:
                    continue

                precio_actual = ventana.iloc[-1]['close']
                precio_open_sig = df.iloc[i+1]['open']

                # Gestión de posición abierta
                if en_posicion:
                    # Revisar si se tocó SL o TP
                    vela_sig = df.iloc[i+1]
                    salida = None
                    motivo = ''

                    if vela_sig['low'] <= sl_precio:
                        salida = sl_precio
                        motivo = 'STOP LOSS'
                    elif vela_sig['high'] >= tp_precio:
                        salida = tp_precio
                        motivo = 'TAKE PROFIT'

                    if salida:
                        pnl = (salida - entrada_precio) * cantidad_btc
                        capital += pnl
                        operaciones.append({
                            'symbol': symbol,
                            'entrada': entrada_precio,
                            'salida': salida,
                            'pnl': pnl,
                            'motivo': motivo,
                            'capital': capital
                        })
                        en_posicion = False

                # Buscar nueva entrada
                if not en_posicion:
                    señal = analizar_señal(ventana, symbol)
                    if señal['accion'] == 'COMPRA' and señal['sl'] and capital > 100:
                        riesgo_usdt = capital * RIESGO_PCT
                        riesgo_por_unidad = precio_open_sig - señal['sl']
                        if riesgo_por_unidad > 0:
                            cantidad_btc   = riesgo_usdt / riesgo_por_unidad
                            entrada_precio = precio_open_sig
                            sl_precio      = señal['sl']
                            tp_precio      = precio_open_sig + (precio_open_sig - señal['sl']) * RATIO_RIESGO
                            en_posicion    = True

            # Cerrar posición abierta al final
            if en_posicion:
                precio_final = df.iloc[-1]['close']
                pnl = (precio_final - entrada_precio) * cantidad_btc
                capital += pnl
                operaciones.append({
                    'symbol': symbol,
                    'entrada': entrada_precio,
                    'salida': precio_final,
                    'pnl': pnl,
                    'motivo': 'FIN_DATOS',
                    'capital': capital
                })

            # Métricas
            total_ops   = len(operaciones)
            if total_ops == 0:
                print(f'  ⚠️  Sin operaciones para {symbol}')
                continue

            ganadoras   = [o for o in operaciones if o['pnl'] > 0]
            perdedoras  = [o for o in operaciones if o['pnl'] <= 0]
            tasa_exito  = len(ganadoras) / total_ops * 100
            pnl_total   = sum(o['pnl'] for o in operaciones)
            pnl_pct     = (pnl_total / CAPITAL_INICIAL) * 100

            # Drawdown máximo
            pico = CAPITAL_INICIAL
            max_dd = 0
            cap_actual = CAPITAL_INICIAL
            for op in operaciones:
                cap_actual = op['capital']
                if cap_actual > pico:
                    pico = cap_actual
                dd = (pico - cap_actual) / pico * 100
                if dd > max_dd:
                    max_dd = dd

            resultado = {
                'symbol':       symbol,
                'operaciones':  total_ops,
                'ganadoras':    len(ganadoras),
                'perdedoras':   len(perdedoras),
                'tasa_exito':   tasa_exito,
                'pnl_total':    pnl_total,
                'pnl_pct':      pnl_pct,
                'drawdown_max': max_dd,
                'capital_final': capital
            }
            resultados_totales.append(resultado)

            # Imprimir reporte
            print(f'\n  ══ REPORTE {symbol} ══')
            print(f'  Total operaciones : {total_ops}')
            print(f'  ✅ Ganadoras       : {len(ganadoras)} ({tasa_exito:.1f}%)')
            print(f'  ❌ Perdedoras      : {len(perdedoras)}')
            print(f'  💰 PnL Total       : ${pnl_total:,.2f} ({pnl_pct:.2f}%)')
            print(f'  📉 Drawdown Máx    : {max_dd:.2f}%')
            print(f'  💵 Capital Final   : ${capital:,.2f}')

        except Exception as e:
            print(f'  ❌ Error en backtest {symbol}: {e}')

    # Guardar resultados en CSV
    if resultados_totales:
        df_res = pd.DataFrame(resultados_totales)
        df_res.to_csv('backtest_resultados.csv', index=False, sep=';')
        print('\n✅ Resultados guardados en backtest_resultados.csv')

    print('\n' + '═'*55)
    return resultados_totales

# ═══════════════════════════════════════════════════════════
#   GUARDAR HISTORIAL
# ═══════════════════════════════════════════════════════════

def guardar_registro(datos):
    df_nuevo = pd.DataFrame([datos])
    if not os.path.isfile(archivo_registro):
        df_nuevo.to_csv(archivo_registro, index=False, sep=';')
    else:
        df_nuevo.to_csv(archivo_registro, mode='a', index=False, header=False, sep=';')

# ═══════════════════════════════════════════════════════════
#   OBTENER BALANCE
# ═══════════════════════════════════════════════════════════

def obtener_balance():
    balance  = exchange.fetch_balance()
    usdt     = balance['free'].get('USDT', 0)
    btc      = balance['total'].get('BTC', 0)
    eth      = balance['total'].get('ETH', 0)
    bnb      = balance['total'].get('BNB', 0)
    sol      = balance['total'].get('SOL', 0)
    return usdt, btc, eth, bnb, sol

# ═══════════════════════════════════════════════════════════
#   CICLO PRINCIPAL
# ═══════════════════════════════════════════════════════════

def ejecutar_bot():
    usdt, btc, eth, bnb, sol = obtener_balance()

    print(f'\n{"═"*45}')
    print(f'  ESTADO DE BILLETERA — JOVATRADING PRO')
    print(f'{"═"*45}')
    print(f'  💵 USDT : ${usdt:,.2f}')
    print(f'  ₿  BTC  : {btc:.6f}')
    print(f'  Ξ  ETH  : {eth:.6f}')
    print(f'  BNB     : {bnb:.6f}')
    print(f'  SOL     : {sol:.6f}')
    print(f'{"═"*45}')

    for symbol in PARES:
        try:
            print(f'\n🔍 Analizando {symbol}...')
            velas = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=300)
            df    = pd.DataFrame(velas, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            df    = calcular_indicadores(df)
            df.dropna(inplace=True)

            if len(df) < 50:
                print(f'  ⚠️  Datos insuficientes para {symbol}')
                continue

            señal  = analizar_señal(df, symbol)
            precio = señal['precio']
            rsi    = df.iloc[-1]['rsi']
            ema200 = df.iloc[-1]['ema200']
            atr    = df.iloc[-1]['atr']
            adx    = df.iloc[-1]['adx']
            macd_h = df.iloc[-1]['macd_hist']

            print(f'  📊 Precio: ${precio:,.4f} | RSI: {rsi:.1f} | ADX: {adx:.1f} | MACD hist: {macd_h:.4f}')
            print(f'  📈 EMA200: ${ema200:,.4f} | ATR: ${atr:,.4f}')
            print(f'  🎯 Señal: {señal["accion"]} (confianza: {señal["confianza"]}/12)')
            if señal['razon']:
                print(f'  📝 {señal["razon"]}')

            accion = señal['accion']
            sl     = señal.get('sl')
            tp     = señal.get('tp')

            # Ejecutar COMPRA con SL y TP
            if accion == 'COMPRA' and sl and tp and usdt > 20:
                riesgo_usdt  = usdt * RIESGO_PCT
                riesgo_por_u = precio - sl
                if riesgo_por_u > 0:
                    cantidad = riesgo_usdt / riesgo_por_u

                    # Redondear según par
                    if 'BTC' in symbol: cantidad = round(cantidad, 5)
                    if 'ETH' in symbol: cantidad = round(cantidad, 4)
                    if 'BNB' in symbol: cantidad = round(cantidad, 3)
                    if 'SOL' in symbol: cantidad = round(cantidad, 2)

                    if cantidad > 0:
                        exchange.create_market_buy_order(symbol, cantidad)
                        print(f'  ✅ COMPRA ejecutada: {cantidad} @ ${precio:,.4f}')
                        print(f'  🛡️ Stop Loss: ${sl:,.4f} | 🎯 Take Profit: ${tp:,.4f}')
                        accion = f'COMPRA_EJECUTADA (SL:{sl:.4f} TP:{tp:.4f})'

            # Guardar registro
            ahora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            guardar_registro({
                'Fecha':     ahora,
                'Symbol':    symbol,
                'Precio':    round(precio, 4),
                'RSI':       round(rsi, 2),
                'ADX':       round(adx, 2),
                'EMA200':    round(ema200, 4),
                'ATR':       round(atr, 4),
                'MACD_hist': round(macd_h, 6),
                'Señal':     accion,
                'Confianza': señal['confianza'],
                'SL':        round(sl, 4) if sl else '',
                'TP':        round(tp, 4) if tp else '',
                'Balance_USDT': round(usdt, 2),
                'Razon':     señal['razon']
            })

            time.sleep(2)  # Pausa entre pares para no saturar la API

        except Exception as e:
            print(f'  ❌ Error en {symbol}: {e}')

# ═══════════════════════════════════════════════════════════
#   INICIO
# ═══════════════════════════════════════════════════════════

print('🚀 JOVATRADING PRO iniciando...')
print(f'   Pares    : {", ".join(PARES)}')
print(f'   Riesgo   : {RIESGO_PCT*100}% por operación')
print(f'   Ratio    : 1:{RATIO_RIESGO} (TP/SL)')
print(f'   Confianza mínima: {MIN_CONFIANZA}/12 indicadores')

print('\n¿Deseas ejecutar el backtesting de 6 meses primero? (s/n): ', end='')
respuesta = input().strip().lower()
if respuesta == 's':
    ejecutar_backtesting()
    print('\n¿Continuar con el bot en vivo? (s/n): ', end='')
    if input().strip().lower() != 's':
        print('Bot detenido.')
        exit()

print('\n▶️  Iniciando bot en vivo...\n')

while True:
    try:
        ejecutar_bot()
    except Exception as e:
        print(f'\n❌ Error general: {e}')
    print(f'\n⏳ Esperando 15 minutos para el próximo análisis...')
    time.sleep(900)  # 15 minutos