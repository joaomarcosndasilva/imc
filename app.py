import streamlit as st
import yfinance as yf
import datetime
from time import sleep

icone_info = "ℹ️"
icone_warning = "⚠️"
icone_erro = "🚨"
icone_sucess = "✅"
hoje = datetime.date.today().strftime("%d/%m/%Y")

def graficos_analises():
    global df
    inicio = st.sidebar.date_input('Data Inicial', datetime.date(2015, 1, 1))
    hoje = st.sidebar.date_input('Data Final', datetime.date.today())
    ticket = yf.Ticker(acao)
    df = ticket.history(period='9y')
    st.line_chart(df.Close)

    st.sidebar.info('GRÁFICOS', icon=icone_sucess)

    cb_volume = st.sidebar.checkbox('Gráfico de Volume')
    if cb_volume:
        with st.spinner('Aguarde...'):
            sleep(0.5)
        st.info('Gráfico de Volume', icon=icone_sucess)
        st.line_chart(df.Volume)

    cb_dividendos = st.sidebar.checkbox('Gráfico de dividendos')
    if cb_dividendos:
        with st.spinner('Aguarde...'):
            sleep(0.5)
        st.info('Gráfico de dividendos!', icon=icone_sucess)
        st.line_chart(df.Dividends)
    st.sidebar.warning('PREVER PREÇOS FUTUROS', icon=icone_warning)

    lr = st.sidebar.checkbox('Regressão Linear')
    if lr:
        with st.spinner('Aguarde...'):
            sleep(1)
        analisar_ativo(codigo_ativo=acao)
            #except:
             #   st.error('A Regressão Linear ainda não está funcionando, por favor, aguarde + alguns dias', icon=icone_erro)

    cb_fbprophet = st.sidebar.checkbox('Prophet (Previsor do Facebook)')
    if cb_fbprophet:
        with st.spinner('Aguarde...'):
            sleep(0.5)
        st.error('O PROPHET ainda não estão funcionando, por favor, aguarde + alguns dias', icon=icone_erro)

    cb_neural = st.sidebar.checkbox('REDE NEURAL')
    if cb_neural:
        with st.spinner('Aguarde...'):
            sleep(0.5)
        st.error('Nenhuma (RN) Rede Neural para previsão ainda, por favor, alguns + alguns dias', icon=icone_erro)


def analisar_ativo(codigo_ativo='CPLE6', periodo_analisado='9'):
    global figura, df
    import pandas as pd
    #import yfinance as yf
    #yf.pdr_override()
    global df, lr, y_de_amanha, df_inicial, x_features, scaler, total, teste, treino, validacao, coeficiente, df2, ativo

    ativo, periodo = codigo_ativo, periodo_analisado,

    #ticket = f'{ativo}.SA'
    #ticketII = yf.Ticker(ticket)
    #df_inicial = ticketII.history(period=f'{periodo}y')
    #ativo = ativo
    df_inicial = df[:]
    df = df_inicial[:]
    df = df.drop(['Dividends', 'Stock Splits'], axis=1)
    df['mm9'] = df['Close'].rolling(9).mean().round(2)
    df['mm21'] = df['Close'].rolling(21).mean().round(2)
    df_inicial = df[:]
    df['Close'] = df['Close'].shift(-1)
    df = df.dropna()

    total = len(df)
    total_inicial = len(df_inicial)

    treino = total - 700
    treino_inicial = total_inicial - 700

    teste = total - 15
    teste_inicial = total_inicial - 15

    ########################################################################################################################
    st.write('A SEPARAÇÃO DOS DADOS SEGUE A SEGUINTE DIVISÃO:')
    st.write(f'\nTreino 0:{treino} - Teste {treino}:{teste} - Validação {teste}:{total}')
    st.write(f'Treino 0:{treino_inicial} - Teste {treino_inicial}:{teste_inicial} - Validação {teste_inicial}:{total_inicial}')
    ########################################################################################################################

    df = df.reset_index()
    df_inicial = df_inicial.reset_index()

    x_features = df.drop(['Date', 'Close'], axis=1)
    x_features_inicial = df_inicial.drop(['Date', 'Close'], axis=1)

    x_features_list = list(x_features)

    y_labels = df['Close']
    y_labels_inicial = df_inicial['Close']

    from sklearn.feature_selection import SelectKBest

    k_best_features = SelectKBest(k='all')
    k_best_features.fit_transform(x_features, y_labels)
    k_best_features_score = k_best_features.scores_
    melhores = zip(x_features_list, k_best_features_score)
    melhores_ordenados = list(reversed(sorted(melhores, key=lambda x: x[1])))

    melhores_variaveis = dict(melhores_ordenados[:15])
    melhores_selecionadas = melhores_variaveis.keys()

    x_features = x_features.drop('Volume', axis=1)
    x_features_inicial = df_inicial.drop(['Date', 'Close', 'Volume'], axis=1)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_features)
    x_features_normalizado = scaler.fit_transform(x_features)

    x_features_inicial = x_features_inicial.dropna()
    scaler.fit(x_features_inicial)
    x_features_inicial_normalizado = scaler.fit_transform(x_features_inicial)

    x_features_normal = pd.DataFrame(x_features_normalizado, columns=list(x_features.columns))
    x_features_normal_inicial = pd.DataFrame(x_features_inicial_normalizado, columns=list(x_features_inicial.columns))

    x_train = x_features_normal[0:treino]
    x_train_inicial = x_features_inicial_normalizado[0:treino_inicial]

    x_test = x_features_normal[treino:teste]
    x_test_inicial = x_features_inicial_normalizado[treino_inicial:teste_inicial]

    y_train = y_labels[0:treino]
    y_train_inicial = y_labels[0:treino_inicial]

    y_test = y_labels[treino:teste]
    y_test_inicial = y_labels[treino_inicial:teste_inicial]


    st.write(f'\nO modelo aprenderá com os dados da linha 0 a {treino} das variáveis {list(x_features.columns)}')
    st.write(f'O modelo testará com os dados da linha {treino} a {teste} da variável Close')
    st.write('\nNa Setunda Parte: ')
    st.write(f'O modelo aprenderá com os dados da linha 0 a {treino_inicial} das variáveis {list(x_features_inicial.columns)}')
    st.write(f'O modelo testará com os dados da linha {treino_inicial} a {teste_inicial} da variável Close')

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_predito = lr.predict(x_test)

    # lr.fit()
    coeficiente = r2_score(y_test, y_predito)

    st.write(f'''O coeficiente é de {coeficiente * 100:.2f}%, isto é,  {coeficiente * 100:.2f}% das variações no valor dopreço futuro de
    Fechamento (Close) é explicada pela variação nas variávies {list(x_features.columns)} do dia anterior''')

    previsao = x_features_normal[teste:total]
    dia = df['Date'][teste:total]
    real = df['Close'][teste:total]

    previsao_hoje = x_features_normal_inicial[teste:total]
    dia_hoje = df_inicial['Date'][teste_inicial:total_inicial]
    real_hoje = df_inicial['Close'][teste_inicial:total_inicial]

    y_pred = lr.predict(previsao)
    y_de_amanha = lr.predict(x_features_inicial_normalizado)

    df2 = pd.DataFrame({'Data': dia, 'Cotacao': real, 'Previsto': y_pred})
    df2['Cotacao'] = df2['Cotacao'].shift(+1)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    figura, ax = plt.subplots(figsize=(16, 8))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_tick_params(rotation=30)

    ax.set_title(f'Teste de Previsão dos ultimos 15 pregões do ativo {ativo.replace(".SA", "")}\nCoeficiente R2 de {round(coeficiente * 100, 2)}% - By J. Brutus', fontsize=24)
    ax.set_ylabel('Preço do ativo em R$', fontsize=14)
    ax.plot(df2['Data'], df2['Cotacao'], marker='o', label='Cotação Real', color='blue')
    ax.plot(df2['Data'], df2['Previsto'], marker='o', label='Cotação Prevista', color='red')

    plt.grid()
    plt.show()
    st.pyplot(figura)
    rodar_nova()


def rodar_nova():
  global x_norm
  import pandas as pd
  import datetime
  import matplotlib.pyplot as plt
  import matplotlib.dates as mdates

  df = df_inicial

  df = df.drop(['Date','Volume' ], axis=1) # vai até o 2237
  df = df.dropna()
  y = df['Close']
  x = df.drop('Close', axis=1)

  scaler.fit(x)
  x_norm = scaler.fit_transform(x)

  y_previsto = lr.predict(x_norm[-1:])
  hoje = datetime.date.today()

  previsao_hoje = pd.DataFrame({'Data':hoje, 'Preco Previsto': y_previsto})

  ###############################################################################


  figura, ax = plt.subplots(figsize=(16, 8))

  ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
  ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
  ax.xaxis.set_tick_params(rotation=30)

  ax.set_title(f'\n\nPrevisão do preço de fechamento para hoje, {hoje.strftime("%d/%m/%Y")} (EM VERDE), do ativo {ativo.replace(".SA", "")}\nCoeficiente R2 de {round(coeficiente * 100, 2)}% - By J. Brutus', fontsize=20)
  ax.set_ylabel('Preço do ativo em R$', fontsize=14)
  ax.plot(df2['Data'], df2['Cotacao'], marker='o', label='Cotação Real', color='blue')
  ax.plot(df2['Data'], df2['Previsto'], marker='o', label='Cotação Prevista', color='red')
  ax.plot(previsao_hoje['Data'], previsao_hoje['Preco Previsto'], marker='o', color='green')

  plt.grid()
  plt.show()
  st.pyplot(figura)


  return

st.title('Análise de ativos da B3 - Versão de teste 1.0')
st.write('by J. Brutus')
st.subheader('Essa aplicação faz uma análise de todos os ativos da B3.')
st.write('Tem muita coisa ainda não funcionando que vou complementá-las com o tempo. A previsão só está funcionando a de Regressão Linear')

st.sidebar.success('ANÁLISES DE ATIVOS', icon=icone_info)

select_modo = st.sidebar.radio("Selecione como você quer ver a análise", ("Lista de ativos", "Digitar o código"))

if select_modo == "Digitar o código":
    acao = st.sidebar.text_input('Digite o código do Ativo e selecione as datas!', 'VALE3', help='Digite o código do ativo sem o ".SA" e pressione ENTER. ')
    acao = f'{acao}.SA'
    if acao:
        try:
            graficos_analises()
        except:
            st.warning(f'Você digitou o ativo {acao}. e selecionou os períodos ')
            st.error("Alguma coisa não está certa. Tente alterar o período de datas")

elif select_modo == "Lista de ativos":
    papeis = ['ABEV3', 'BBAS3', 'BBDC4', 'PETR4', 'VALE3', 'BEEF3', 'CMIG4', 'CPLE3']
    acao = st.sidebar.selectbox('Selecione o ativo e as datas que preferir', papeis, help="Está Lista contém apenas os ativos de ações mais líquidas do índice Ibovespa.")
    acao = f'{acao}.SA'
    graficos_analises()

else:
    st.info('Marque como você vai querer a análise')

