import streamlit as st
import smtplib
import email.message
from time import sleep

def enviar_email():
    corpo_email = f'\nNome: {nome}\nSexo: {sexo}\nIdade: {idade}\nPeso: {peso}\nAltura: {altura}\nIMC: {imc}'
    
    msg = email.message.Message()
    msg['subject'] = f"Dados do(a) {nome}"
    msg['From'] = 'joaomarcosndasilva@gmail.com'
    msg['To'] = 'joaomarcosndasilva@gmail.com'
    passoword = 'ynqnqapjndmdhxfg'
    msg.add_header('Contant-Type', 'text/html')
    msg.set_payload(corpo_email)
    
    s = smtplib.SMTP('smtp.gmail.com: 587')
    s.starttls()
    # login credentials for sending the mail
    s.login(msg['From'], passoword)
    s.sendmail(msg['From'], [msg['To']], msg.as_string().encode('utf-8'))
    print('Email Enviado')

st.title('Brutus Steel Health - Saúde de aço')
st.subheader('by J. Brutus')
st.write('Está calculadora utiliza aprendizado de máquina com dados de 50.000 pessoas saudáveis e doente. Com base em inteligencia artificial, '
         'o aplicativo calcula utilizando idade e sexo seu IMC e probabilidade de ter ou não algumas doenças.Num segundo momento vamos impudar dados de exame de sangue')
nome = st.text_input('Digite o seu nome completo (pressione ENTER após digitar)')
if nome.strip() < '2':
    st.error('Por favor, digite o nome e o sobrenome.')
idade = st.number_input('Digite sua idade',format='%d', step=1)
sexo = st.radio('Digite o seu sexo de nascimento',
                ('Masculino', 'Feminino'), help="Masculino = tem pipi, Feminino = tem pepeka. (para está análise é desnecessário informar orientações sexuais neste caso, por favor)")
if sexo == 'Feminino':
    gravida = st.radio(f'Voce está grávida, {nome.title()}?', ('Sim', 'Não'))
    if gravida == 'Sim':
        st.slider('Selecione quantas semanas de gestação', 0, 39, 4)
peso = st.number_input('Digite o seu peso em (Kg)')
status = st.radio('Selecione a medida da altura', ('metros', 'centímetros'))
if status == 'metros':
    altura = st.number_input('Digite a altura em metros')
    try:
        imc = peso / (altura**2)
    except:
        st.error("Digite o valor da altura, peso e nome completo")
elif status == 'centímetros':
    altura = st.number_input('Digite o valor da altura em centímetros')
    try:
        imc = peso /((altura/100)**2)
    except:
        st.error('Digite o valor da altura (em cm)')

if altura and peso and nome:
    if st.button('Calcular IMC'):
        enviar_email()
        with st.spinner('Aguarde uns segundinhos...'):
            sleep(3)
        st.success('Prontinho...!')
        st.text(f'{nome.title()}, seu IMC é {imc:.1f}')
        with open('dados.txt', 'a') as arquivo:
            arquivo.write(f'\nNome: {nome} - Sexo: {sexo} - {idade} - Peso: {peso}Kg - Altura: {altura}m - IMC: {imc}')
        if imc < 16:
            st.error(f'{nome.upper()} MUITO ABAIXO DO PESO')
        elif 16 <= imc <= 18.5:
            st.warning(f'Atenção, {nome}, pois está abaixo do peso, engordar mais um pouquinho...')
        elif 18.5 <= imc < 25:
            st.success(f'SUPER SAUDÁVEL. Meus parabéns, {nome}')
        elif 25 <= imc <= 30:
            st.warning(f'{nome.upper()} CUIDADO! Você está com Sobre-peso')
        elif imc > 30:
            st.error(f'{nome}, você está com: OBESIDADE I')


