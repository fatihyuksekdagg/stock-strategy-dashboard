#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install plotly dash dash-bootstrap-components')
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score,f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout,Input
from tensorflow.keras.optimizers import RMSprop
import yfinance as yf
import matplotlib.pyplot as plt


# In[2]:


# Apple'ın verilerini yfinance ile alma
data = yf.download('AAPL', start='2018-05-15', end='2024-05-17')

# Tarih formatını eşleştirme
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date']).dt.date


# In[3]:


# Veriyi yfinance ile çekme
data = yf.download('AAPL', start='2020-05-15', end='2024-05-17')
data.reset_index(inplace=True)

# Figure nesnesini oluşturma
fig = go.Figure()

# Candlestick grafiğini figüre ekleme
fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name='Fiyat'))

# Grafiğin layout ayarlarını yapma
fig.update_layout(
    title='AAPL Hisse Fiyatı',
    xaxis_title='Tarih',
    yaxis_title='Fiyat (USD)',
    margin=dict(l=20, r=20, t=50, b=20)
)

# Grafiği gösterme (Jupyter Notebook için)
fig.show()


# In[4]:


# Teknik göstergeleri hesaplama
def MACD(data, period_short=12, period_long=26, period_signal=9):
    ShortEMA = data['Close'].ewm(span=period_short, adjust=False).mean()
    LongEMA = data['Close'].ewm(span=period_long, adjust=False).mean()
    data['MACD'] = ShortEMA - LongEMA
    data['Signal_Line'] = data['MACD'].ewm(span=period_signal, adjust=False).mean()

def RSI(data, period=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))

def Stochastic(data, period=14):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    data['%K'] = (data['Close'] - low_min) * 100 / (high_max - low_min)
    data['%D'] = data['%K'].rolling(window=3).mean()

def Momentum(data, period=10):
    data['Momentum'] = data['Close'] - data['Close'].shift(period)


# In[5]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf

app = dash.Dash(__name__)

# Kullanıcı Arayüzü Layout'u
app.layout = html.Div([
    html.H1("Stock Analysis with Technical Indicators"),
    
    dcc.Dropdown(
        id='indicator-dropdown',
        options=[
            {'label': 'MACD', 'value': 'MACD'},
            {'label': 'RSI', 'value': 'RSI'},
            {'label': 'Stochastic', 'value': 'Stochastic'},
            {'label': 'Momentum', 'value': 'Momentum'}
        ],
        value='MACD'
    ),
    
    dcc.Graph(id='stock-graph'),
    
])

# Grafikleri güncellemek için callback fonksiyonu
@app.callback(
    Output('stock-graph', 'figure'),
    Input('indicator-dropdown', 'value')
)
def update_graph(selected_indicator):
    # Veriyi yfinance ile çek
    data = yf.download('AAPL', start='2020-05-15', end='2024-05-17')
    data.reset_index(inplace=True)
    
    # Figure nesnesi oluştur
    fig = go.Figure()
    
    if selected_indicator == 'MACD':
        ShortEMA = data['Close'].ewm(span=12, adjust=False).mean()
        LongEMA = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ShortEMA - LongEMA
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], mode='lines', name='Signal Line'))
        
    elif selected_indicator == 'RSI':
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        data['RSI'] = 100 - (100 / (1 + RS))
        
        fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'))
        fig.add_shape(type="line", x0=data['Date'].min(), x1=data['Date'].max(), y0=30, y1=30, line=dict(color="Red", width=2, dash="dashdot"))
        fig.add_shape(type="line", x0=data['Date'].min(), x1=data['Date'].max(), y0=70, y1=70, line=dict(color="Red", width=2, dash="dashdot"))
        
    elif selected_indicator == 'Stochastic':
        low_min = data['Low'].rolling(window=14).min()
        high_max = data['High'].rolling(window=14).max()
        data['%K'] = (data['Close'] - low_min) * 100 / (high_max - low_min)
        data['%D'] = data['%K'].rolling(window=3).mean()
        
        fig.add_trace(go.Scatter(x=data['Date'], y=data['%K'], mode='lines', name='%K'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['%D'], mode='lines', name='%D'))
        fig.add_shape(type="line", x0=data['Date'].min(), x1=data['Date'].max(), y0=20, y1=20, line=dict(color="Green", width=2, dash="dashdot"))
        fig.add_shape(type="line", x0=data['Date'].min(), x1=data['Date'].max(), y0=80, y1=80, line=dict(color="Green", width=2, dash="dashdot"))
        
    elif selected_indicator == 'Momentum':
        data['Momentum'] = data['Close'] - data['Close'].shift(10)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Momentum'], mode='lines', name='Momentum'))
    
    fig.update_layout(
        title=f"{selected_indicator} Indicator for AAPL",
        xaxis_title='Date',
        yaxis_title='Indicator Value',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)


# In[6]:


#Tarih formatını eşleştirme ve indeks olarak ayarlama
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Göstergeleri hesapla
MACD(data)
RSI(data)
Stochastic(data)
Momentum(data)
print(data[['MACD', 'Signal_Line', 'RSI', '%K', '%D', 'Momentum']].tail(10))


# In[7]:


# Göstergelerin kombinasyonu
def combined_indicators(data):
    data['Combined_Score'] = (
        (data['MACD'] > data['Signal_Line']).astype(int) +
        (data['RSI'] < 30).astype(int) -
        (data['RSI'] > 70).astype(int) +
        (data['%K'] < 20).astype(int) -
        (data['%K'] > 80).astype(int) +
        (data['Momentum'] > 0).astype(int)
    )
    data['Signal'] = np.where(data['Combined_Score'] > 0, 1, 0)
    data['Signal'] = np.where(data['Combined_Score'] < 0, -1, data['Signal'])

combined_indicators(data)


# In[6]:


# NaN değerleri olan satırları kaldır
data.dropna(inplace=True)


# In[10]:


# Veriyi ölçeklendir

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal_Line', 'RSI', '%K', '%D', 'Momentum']])


# In[11]:


# Veriyi hazırla
def prepare_data(data, window_size=120):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 3])  # Close sütunu genellikle index 3'te yer alır
    return np.array(X), np.array(y)


# In[12]:


# Veriyi diziye dönüştür ve eğitim/test olarak ayır
window_size = 120            
X, y = prepare_data(scaled_data, window_size)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Veriyi eğitim, doğrulama ve test setlerine ayırma
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#windows size kısmı: bunu mesela aşağıda last 60 days kısmı gbi düşün yazdırmak istediğimiz kadarolacak örneğin son 2 hafta yazdırackasak buraya 14 yazdıracağız
#validation dataset test


# In[13]:


# Veriyi eğitim ve test setlerine ayırma
#split = int(0.9 * len(X))
#X_train, X_test = X[:split], X[split:]
#y_train, y_test = y[:split], y[split:]


# In[14]:


# Modeli tanımla ve eğit
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
    return model


# In[15]:


#RNN Model
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
    return model


# In[16]:


callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ModelCheckpoint('best_lstm_model.keras', save_best_only=True)
]


# In[17]:


# LSTM modelini oluştur ve eğit
lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
lstm_history = lstm_model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.2, callbacks=callbacks)


# In[18]:


# RNN Modelini oluştur ve eğit
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ModelCheckpoint('best_rnn_model.keras', save_best_only=True)
]


# In[19]:


rnn_model = build_rnn_model((X_train.shape[1], X_train.shape[2]))
rnn_history = rnn_model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.2, callbacks=callbacks)


# In[20]:


# Karesel Hata ve Accuracy değerlerini yazdır
mse_lstm = lstm_model.evaluate(X_test, y_test)
mse_rnn = rnn_model.evaluate(X_test, y_test)
predictions_lstm = lstm_model.predict(X_test)
predictions_rnn = rnn_model.predict(X_test)

print("LSTM Mean Squared Error:", mse_lstm)
print("RNN Mean Squared Error:", mse_rnn)


# In[21]:


# Model performansını doğrulama seti üzerinde değerlendirme
val_mse = lstm_model.evaluate(X_val, y_val)
print(f"Validation MSE: {val_mse}")


# In[22]:


# Sadece Close fiyatını geri ölçeklendirme
def inverse_transform_close(scaler, data, column_index):
    inverse_data = np.zeros((len(data), scaler.n_features_in_))
    inverse_data[:, column_index] = data[:, 0]
    return scaler.inverse_transform(inverse_data)[:, column_index]


# In[23]:


predicted_prices_lstm = predictions_lstm.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

predicted_prices_lstm = inverse_transform_close(scaler, predicted_prices_lstm, 3)
actual_prices = inverse_transform_close(scaler, y_test_reshaped, 3)

# Adjusted inverse transform
predicted_prices_rnn = predictions_rnn.reshape(-1, 1)
actual_prices = y_test.reshape(-1, 1)
predicted_prices_rnn = inverse_transform_close(scaler, predicted_prices_rnn, 3)
actual_prices = inverse_transform_close(scaler, actual_prices, 3)


# In[24]:


plt.figure(figsize=(14, 7))
plt.plot(actual_prices, color='blue', label='Actual AAPL Price')
plt.plot(predicted_prices_lstm, color='red', label='Predicted AAPL Price (LSTM)')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()


# Plot predictions
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, color='blue', label='Actual AAPL Price')
plt.plot(predicted_prices_rnn, color='red', label='Predicted AAPL Price (RNN)')
plt.title('AAPL Stock Price Prediction (RNN)')
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()


# In[25]:


# Stop loss ve take profit seviyeleri
stop_loss_threshold = 0.02  # %2 kayıp
take_profit_threshold = 0.03  # %3 kazanç
initial_capital = 10000  # Başlangıç sermayesi
shares = 10  # Alınacak hisse adedi

def execute_trades(predicted_prices, actual_prices, initial_capital, shares, stop_loss_threshold, take_profit_threshold):
    capital = initial_capital
    initial_price = actual_prices[0]
    
    for i in range(1, len(predicted_prices)):
        current_price = actual_prices[i]
        predicted_price = predicted_prices[i]
        
        # Mevcut sermaye ve hisse adedi ile işlem yapma
        current_value = capital
        shares_value = shares * current_price
        portfolio_value = current_value + shares_value

        # Stop loss
        if (current_price - initial_price) / initial_price <= -stop_loss_threshold:
            capital -= shares_value
            print(f"Stop loss triggered at {current_price}. Remaining capital: {capital}")
            break

        # Take profit
        if (current_price - initial_price) / initial_price >= take_profit_threshold:
            capital += shares_value
            print(f"Take profit triggered at {current_price}. Total capital: {capital}")
            break

        # Güncellenmiş sermaye
        capital = portfolio_value
        initial_price = current_price

    return capital

# Model tahminleri ve sermaye hesaplamaları
final_capital_lstm = execute_trades(predicted_prices_lstm, actual_prices, initial_capital, shares, stop_loss_threshold, take_profit_threshold)
final_capital_rnn = execute_trades(predicted_prices_rnn, actual_prices, initial_capital, shares, stop_loss_threshold, take_profit_threshold)

print("LSTM Final Capital:", final_capital_lstm)

print("RNN Final Capital:", final_capital_rnn)


# In[26]:


# F1 Skoru Hesaplama ve Yazdırma
def calculate_f1_score(predicted, actual, threshold=0.01):
    predicted_signals = np.where(predicted > actual * (1 + threshold), 1, np.where(predicted < actual * (1 - threshold), -1, 0))
    actual_signals = np.where(actual[1:] > actual[:-1], 1, np.where(actual[1:] < actual[:-1], -1, 0))
    actual_signals = np.insert(actual_signals, 0, 0)  # İlk değeri nötr olarak ekle
    f1 = f1_score(actual_signals[:-1], predicted_signals[1:], average='macro', zero_division=1)
    return f1


# In[27]:


f1_lstm = calculate_f1_score(predicted_prices_lstm, actual_prices)
f1_rnn = f1_score(np.sign(actual_prices[1:] - actual_prices[:-1]), np.sign(predicted_prices_rnn[1:] - predicted_prices_rnn[:-1]), average='macro', zero_division=1)

print("LSTM F1 Score:", f1_lstm)

# Calculate F1 Score and Accuracy for RNN

print("RNN F1 Score:", f1_rnn)



# In[28]:


# Accuracy Hesaplama ve Yazdırma
def calculate_accuracy(predicted, actual, threshold=0.01):
    predicted_signals = np.where(predicted > actual * (1 + threshold), 1, np.where(predicted < actual * (1 - threshold), -1, 0))
    actual_signals = np.where(actual[1:] > actual[:-1], 1, np.where(actual[1:] < actual[:-1], -1, 0))
    actual_signals = np.insert(actual_signals, 0, 0)  # İlk değeri nötr olarak ekle
    accuracy = accuracy_score(actual_signals[:-1], predicted_signals[1:])
    return accuracy

accuracy_lstm = calculate_accuracy(predicted_prices_lstm, actual_prices)
accuracy_rnn = accuracy_score(np.sign(actual_prices[1:] - actual_prices[:-1]), np.sign(predicted_prices_rnn[1:] - predicted_prices_rnn[:-1]))

print("LSTM Accuracy:", accuracy_lstm)
print("RNN Accuracy:", accuracy_rnn)


# In[29]:


# Gelecek günleri tahmin etme fonksiyonu
def predict_next_days(model, previous_data, days=2):
    next_days = []
    last_window = previous_data[-window_size:]
    
    for _ in range(days):
        X_pred = last_window.reshape((1, window_size, scaled_data.shape[1]))
        pred_price = model.predict(X_pred)
        
        # Dummy array for inverse_transform
        pred_full = np.zeros((1, scaled_data.shape[1]))
        pred_full[0, 3] = pred_price  # 'Close' fiyatı için tahmin
        
        # Update last_window with the predicted price
        last_window = np.append(last_window[1:], pred_full, axis=0)
        
        # Store the predicted price
        next_days.append(pred_full[0, 3])

    # Return the inverse transformed prices
    next_days = np.array(next_days).reshape(-1, 1)
    dummy = np.zeros((len(next_days), scaled_data.shape[1]))
    dummy[:, 3] = next_days[:, 0]  # 'Close' sütunu
    return scaler.inverse_transform(dummy)[:, 3]

# Gelecek 2 günü tahmin etme
next_2_days_lstm = predict_next_days(lstm_model, scaled_data, 2)
print("LSTM Next 2 Days Predictions:", next_2_days_lstm)

next_2_days_rnn = predict_next_days(rnn_model, scaled_data, 2)
print("RNN Next 2 Days Predictions:", next_2_days_rnn)


# In[8]:


###############  P/E  ROE ORANLARI #########

# Apple'ın temel istatistiklerini ve mali bilgilerini yfinance ile çekme
ticker = yf.Ticker("AAPL")

# Temel istatistikleri ve mali bilgileri çekme
info = ticker.info
financials = ticker.financials
balancesheet = ticker.balance_sheet

# EPS ve Net Income değerlerini çekme
eps = info['trailingEps'] if 'trailingEps' in info else None  # Son dönem EPS değeri

# Mali bilgilerden Net Income değerini çekme
if 'Net Income' in financials.index:
    net_income = financials.loc['Net Income'].iloc[0]
elif 'Net Income Common Stockholders' in financials.index:
    net_income = financials.loc['Net Income Common Stockholders'].iloc[0]
else:
    net_income = None

# Bilançodan Total Stockholder Equity değerini çekme
if 'Total Stockholder Equity' in balancesheet.index:
    total_equity = balancesheet.loc['Total Stockholder Equity'].iloc[0]
elif 'Stockholders Equity' in balancesheet.index:
    total_equity = balancesheet.loc['Stockholders Equity'].iloc[0]
else:
    total_equity = None

# Şirketin güncel kapanış fiyatını almak için son veriyi çekme
data = yf.download('AAPL', start='2024-05-01', end='2024-05-14')
latest_price = data['Close'].iloc[-1]

# P/E Oranını Hesaplama
if eps:  # EPS bilgisi varsa P/E oranını hesapla
    pe_ratio_real_time = latest_price / eps
    print(f"Gerçek Zamanlı P/E Oranı: {pe_ratio_real_time:.2f}")
else:
    print("EPS bilgisi bulunamadı.")

# P/E Oranına Göre Yorum Yapma
if eps:
    if pe_ratio_real_time < 10:
        print(f"P/E oranı {pe_ratio_real_time:.2f}, çok düşük. Şirketin hisseleri düşük değerlendirilmiş olabilir, potansiyel bir değer yatırımı olabilir.")
    elif pe_ratio_real_time < 15:
        print(f"P/E oranı {pe_ratio_real_time:.2f}, düşük. Şirketin hisseleri makul bir değerlemeye sahip, orta düzeyde büyüme bekleniyor.")
    elif pe_ratio_real_time < 25:
        print(f"P/E oranı {pe_ratio_real_time:.2f}, orta. Şirketin hisseleri adil bir değerlemeye sahip, ortalama büyüme oranı bekleniyor.")
    elif pe_ratio_real_time < 40:
        print(f"P/E oranı {pe_ratio_real_time:.2f}, yüksek. Şirketin hisseleri yüksek değerlendirilmiş, yüksek büyüme beklentileri var ama aşırı değerlenme riski de taşıyor.")
    else:
        print(f"P/E oranı {pe_ratio_real_time:.2f}, çok yüksek. Şirketin hisseleri aşırı değerlendirilmiş, spekülatif bir bölgede ve yüksek fiyat düzeltmeleri riski barındırıyor.")

# ROE Hesaplama
if net_income and total_equity:  # Net Income ve Total Equity bilgisi varsa ROE hesapla
    roe_real_time = (net_income / total_equity) * 100  # Yüzde (%) olarak hesapla
    print(f"Gerçek Zamanlı ROE: {roe_real_time:.2f}%")

    # ROE'ye Göre Yorum Yapma
    if roe_real_time < 5:
        print(f"ROE {roe_real_time:.2f}%, çok düşük. Şirket sermayesini etkili kullanmıyor, finansal zorluklar veya düşük karlılık sektörleri işaret edebilir.")
    elif roe_real_time < 10:
        print(f"ROE {roe_real_time:.2f}%, düşük. Şirket sermayesini ortalama düzeyde kullanıyor, bazı operasyonel verimsizlikler olabilir.")
    elif roe_real_time < 20:
        print(f"ROE {roe_real_time:.2f}%, orta. Şirket sermayesini iyi kullanıyor, sağlıklı bir oranda kar elde ediyor.")
    elif roe_real_time < 40:
        print(f"ROE {roe_real_time:.2f}%, yüksek. Şirket sermayesini çok etkili kullanarak yüksek karlılık elde ediyor.")
    else:
        print(f"ROE {roe_real_time:.2f}%, çok yüksek. Olağanüstü yüksek karlılık gösteriyor, ancak yüksek borç seviyeleri veya riskler barındırabilir.")
else:
    print("Net Income veya Total Equity bilgisi bulunamadı.")


# In[ ]:





# In[ ]:





# In[9]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Dash uygulamasını başlat
app = dash.Dash(__name__)

# Uygulama layoutunu ayarlama
app.layout = html.Div([
    html.H1("Stock Analysis Dashboard"),
    
    # Performans metriklerini göstermek için bir bölüm
    html.Div([
        html.Div([
            html.H3("MSE"),
            html.P(id='mse-value', children="")
        ], className='metric-box'),
        
        html.Div([
            html.H3("F1 Skoru"),
            html.P(id='f1-value', children="")
        ], className='metric-box'),
        
        html.Div([
            html.H3("Doğruluk"),
            html.P(id='accuracy-value', children="")
        ], className='metric-box')
    ], className='metrics-container', style={'display': 'flex', 'justify-content': 'space-around', 'margin': '20px'}),
    
    # Grafikler ve diğer UI elementleriniz burada yer alabilir
    dcc.Graph(id='stock-graph'),
    html.Button('Update Metrics', id='update-metrics-button'),
    
    # Modelin tahminlerini ve gerçek değerleri göstermek için bir grafik
    dcc.Graph(id='prediction-graph')
])

# Stil ayarları
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


# In[10]:


@app.callback(
    [Output('mse-value', 'children'),
     Output('f1-value', 'children'),
     Output('accuracy-value', 'children')],
    [Input('update-metrics-button', 'n_clicks')]
)
def update_metrics(n_clicks):
    # Metriklerinizi hesaplayan kısımlar burada yer alacak
    # Örnek olarak sabit değerlerle başlayabiliriz:
    
    # LSTM ve RNN için önceden hesaplanmış metrikleri kullanın
    mse_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)
    f1_lstm = calculate_f1_score(predicted_prices_lstm, actual_prices)
    accuracy_lstm = calculate_accuracy(predicted_prices_lstm, actual_prices)
    
    # Metrikleri yuvarlayarak göster
    return f"{mse_lstm:.4f}", f"{f1_lstm:.4f}", f"{accuracy_lstm:.4%}"


# In[11]:


@app.callback(
    Output('prediction-graph', 'figure'),
    [Input('update-metrics-button', 'n_clicks')]
)
def update_predictions(n_clicks):
    # Tahminleri ve gerçek fiyatları gösteren bir grafik oluştur
    fig = go.Figure()
    
    # Gerçek fiyatları çiz
    fig.add_trace(go.Scatter(x=data.index[-len(actual_prices):], y=actual_prices, mode='lines', name='Actual'))
    
    # LSTM ve RNN tahminlerini çiz
    fig.add_trace(go.Scatter(x=data.index[-len(predicted_prices_lstm):], y=predicted_prices_lstm.flatten(), mode='lines', name='LSTM Predictions'))
    fig.add_trace(go.Scatter(x=data.index[-len(predicted_prices_rnn):], y=predicted_prices_rnn.flatten(), mode='lines', name='RNN Predictions'))
    
    # Grafik başlığı ve eksen isimleri
    fig.update_layout(title='Stock Price Predictions',
                      xaxis_title='Date',
                      yaxis_title='Stock Price (USD)',
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


# In[12]:


if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:





# In[ ]:





# In[35]:


#son 60 gün ywerine son 30 gün veya son 2 hafta olsa tarzında bir güncelleme


# In[ ]:





# In[36]:


#results kısmı raporda artacak detaylı


# In[37]:


#transformers model?? attention layer


# In[13]:


#learning rate değişimi dene


# In[14]:


#bir kaç tane şirketin çıktısını hazdırcaz tablo olarak


# In[ ]:




