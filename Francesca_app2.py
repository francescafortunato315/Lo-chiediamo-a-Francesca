import streamlit as st
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import json
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


def costruisci_risposta(data):
    if 'Mi dispiace' not in data['answer']:
        #risp = data['answer'].split('\n')[0] +'\n'
        risp = data['answer'] + '\n'
        for doc in data['context']:
            meta = doc.metadata
            nome = meta.get('nome_proprio', 'Nome non disponibile')
            genere = meta.get('genere', 'Genere non specificato')
            colore = meta.get('colore', 'Colore non specificato')
            if isinstance(colore, list):
                colore = " o ".join(colore)
            materiale = meta.get('materiale', 'Materiale non specificato')
            if isinstance(materiale, list):
                materiale = " e ".join(materiale)
            taglie = ", ".join(meta.get('taglie disponibili', meta.get('taglie', [])))
            stile = meta.get('stile', 'Stile non specificato')
            if isinstance(stile, list):
                stile = ", ".join(stile)
            occasioni = ", ".join(meta.get("occasioni d'uso", meta.get('occasioni', [])))
            link = meta.get('link', '#')

            # Formattazione della risposta
            risp += f"\n**{nome}**\n"
            risp += f"- Genere: {genere}\n"
            risp += f"- Colore: {colore}\n"
            risp += f"- Materiale: {materiale}\n"
            risp += f"- Stile: {stile}\n"
            risp += f"- Taglie disponibili: {taglie}\n"
            risp += f"- Occasioni d'uso: {occasioni}\n"
            risp += f"- Link: [Visualizza il prodotto]({link})\n"
    else:
        risp = data['answer']
    return risp



def reset_chat():
    st.session_state.messages = []

    if "initialized" in st.session_state:
        del st.session_state["initialized"]

    if "user_input" in st.session_state:
        del st.session_state["user_input"]
        st.session_state.user_input = None

def carica_profilo(nome_utente):
    filename = f'profilo_{nome_utente}.txt'
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
            profilo = file.read()
        return profilo
    except FileNotFoundError:
        return st.write("Profilo cliente non trovato.")


# hashed_passwords = Hasher(['abc', 'def']).generate()

with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    cookie_key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days'],
)
authenticator.login()

os.environ["OPENAI_API_KEY"] = st.secrets['api_key']
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

with open('catalogo_aggiornato.json', 'r') as file:
    catalogo = json.load(file)

# Caricamento vector store
vector_store = FAISS.load_local('vector_store.faiss', embeddings, allow_dangerous_deserialization=True)

# Inizializza il modello LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)




def inizializza_stato():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'user_input' not in st.session_state:
        st.session_state.user_input = None


if "initialized" not in st.session_state:
    inizializza_stato()
    st.session_state.initialized = True  # Flag per evitare ripetizioni

if st.session_state["authentication_status"] is False:
    ## ログイン成功ログイン失敗
    st.error('Username/password is incorrect')

elif st.session_state["authentication_status"] is None:
    ## デフォルト
    st.warning('Please enter your username and password')

elif st.session_state["authentication_status"]:

    # Carica il profilo dell'utente loggato
    name = config['credentials']['usernames'][st.session_state["username"]]['name']
    profilo_cliente = carica_profilo(name)
    # st.write(profilo_cliente)

    # Prompt per contestualizzare le domande
    contextualize_q_system_prompt = (
        "Dato uno storico della conversazione e l'ultima domanda dell'utente,"
        "che potrebbe fare riferimento al contesto nella cronologia della chat,"
        "formula una domanda autonoma che possa essere compresa senza fare riferimento "
        "allo storico della conversazione. NON rispondere alla domanda, limitati a "
        "riformularla se necessario; altrimenti, restituiscila così com'è."
    )

    # Definisci il contesto e i prompt per Langchain
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Creazione del retriever contestuale
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(), contextualize_q_prompt
    )

    # Prompt del sistema per l'assistente
    system_prompt = (
        "Sei un assistente virtuale esperto nell'aiutare gli utenti a trovare capi di abbigliamento che rispondano alle loro esigenze e preferenze personali. "
        "Quando rispondi, considera prioritariamente le informazioni nel profilo del cliente (ad esempio, genere, stile preferito, acquisti o ricerche precedenti) "
        "per identificare i prodotti più adatti nel catalogo.\n\n"

        "**Compiti principali:**\n"
        "1. Analizza i dati forniti nel contesto recuperato dal database vettoriale e seleziona esclusivamente i prodotti che soddisfano i seguenti criteri:\n"
        "   - Genere corrispondente.\n"
        "   - Stile preferito.\n"
        "   - Eventuali altre preferenze espresse nel profilo del cliente o nella richiesta specifica.\n\n"

        "2. Restituisci una lista dei prodotti più pertinenti, includendo per ciascuno: \n"
        "   - Nome del prodotto (`nome_proprio`)\n"
        "   - Link al prodotto (`link`)\n"
        "   - Eventuali altre informazioni brevi che possano aiutare l'utente a capire la compatibilità del prodotto con le sue esigenze (ad esempio, taglie disponibili, colori, materiali).\n\n"

        "3. I prodotti devono essere ordinati per rilevanza rispetto alle preferenze del cliente e presentati in modo chiaro, conciso e facilmente comprensibile. "
        "Non includere dettagli tecnici come materiali, taglie o descrizioni lunghe, ma concentrati solo sull'essenziale per orientare il cliente verso la scelta giusta.\n\n"

        "4. Se non trovi prodotti che soddisfano i criteri del cliente, rispondi in modo cortese, iniziando con 'Mi dispiace!' e spiegando che non ci sono prodotti adatti alle specifiche esigenze.\n\n"

        "**Contesto fornito:**\n"
        "- **Profilo Cliente:**\n"
        "{profilo_cliente}\n\n"
        "- **Context (risultati dal database vettoriale):**\n"
        "{context}\n"
    )

    # Definisci i prompt per Langchain
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )



    # Creazione della chain di Q&A
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Creazione della chain di retrieval con memoria storica
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    # Interfaccia della chat
    # st.title("Assistente Virtuale per Acquisti di Abbigliamento")
    # st.write("Chatta con l'assistente per trovare i prodotti adatti alle tue esigenze di stile.")

    # Visualizza il profilo cliente
    with st.sidebar:
        st.image('icon_iniziale.jpeg', width=200)
        name = config['credentials']['usernames'][st.session_state["username"]]['name']
        st.markdown(f'### Ciao {name}! Il tuo *personal shopper* ti da il benvenuto!')

        st.divider()
        st.write("Oppure usa uno di questi suggerimenti:")

        if st.button("Scopri le ultime novità della settimana"):
            user_input = "Quali sono gli ultimi articoli caricati a catalogo?"
            st.session_state.user_input = user_input
            st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "user_icon.png"})
            st.session_state.chat_history.append(HumanMessage(content=user_input))

        if st.button("Cerco un maxi maglione per la montagna"):
            user_input = "Cerco un maxi maglione per la montagna"
            st.session_state.user_input = user_input
            st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "user_icon.png"})
            st.session_state.chat_history.append(HumanMessage(content=user_input))

        if st.button("Qual è l'outfit più indicato per una cena elegante?"):
            user_input = "Cerco un abito elegante"
            st.session_state.user_input = user_input
            st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "user_icon.png"})
            st.session_state.chat_history.append(HumanMessage(content=user_input))

        st.divider()

        if authenticator.logout('Logout', 'sidebar'):
            reset_chat()

        if st.button("Riparti con una nuova richiesta"):
            reset_chat()

        # Gestione dell'interazione dell'utente tramite chat
    st.header('Lo chiediamo a Francesca')
    st.divider()

    with st.chat_message("assistant", avatar='assistant_icon.png'):
        st.write(f"Ciao {name}! Come posso aiutarti oggi?")

    # Mostra la cronologia della chat
    for message in st.session_state.messages:
        if message['role'] == 'assistant':
            with st.chat_message('assistant', avatar='assistant_icon.png'):
                st.write(message['content'])
        else:
            with st.chat_message('user', avatar='user_icon.png'):
                st.write(message['content'])

    # st.sidebar.header("Profilo Cliente")
    # st.sidebar.write(profilo_cliente)

    # Mantieni la cronologia della chat in Streamlit session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Campo di input per la chat
    # user_input = st.chat_input("Cosa stai cercando?")

    # Esegui solo se è stato inserito un input
    # if user_input := st.chat_input('Cosa stai cercando?'):
    if user_input := st.chat_input('Fammi la tua domanda'):
        st.session_state.user_input = user_input
        st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "user_icon.png"})

        st.session_state.chat_history.append(HumanMessage(content=user_input))

        with st.chat_message("user", avatar="user_icon.png"):
            st.write(user_input)

    if st.session_state.user_input:
        with st.spinner("Sto elaborando la tua richiesta..."):
            # Invoca la chain RAG con la domanda e la cronologia della chat
            response = rag_chain.invoke(
                {"input": user_input, "chat_history": st.session_state.chat_history, 'profilo_cliente': profilo_cliente}
            )

            #response_formatted = costruisci_risposta(response)
            # Aggiungi la risposta dell'assistente alla cronologia
            st.session_state.chat_history.append(AIMessage(content=response["answer"]))
            #st.session_state.chat_history.append(AIMessage(content=response_formatted))
            # Mostra il messaggio dell'assistente
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"], "avatar": 'assistant_icon.png'}
            )

            with st.chat_message('assistant', avatar='assistant_icon.png'):
                st.write(response["answer"])

    # Visualizza la cronologia della chat
    # st.subheader("Cronologia della Chat")
    # for message in st.session_state.chat_history:
    # if isinstance(message, HumanMessage):
    # st.write(f"**Tu**: {message.content}")
    # elif isinstance(message, AIMessage):
    # st.write(f"**Assistente**: {message.content}")

