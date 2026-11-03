import streamlit as st
import numpy as np
from storage import save_simulation, load_simulation
#configurarea ferestrei din browser, prima instructiune streamlit din fisier
st.set_page_config(
    page_title="CelLab - Cellular Automata",
    layout="wide"
)

with st.sidebar:
    st.header("⚙️ Setări Simulare")

    #alegerea modelului: dropdown/selectbox
    tip_automat=st.selectbox(
        "Alege Automatul Celular:",
        ["Game of Life", "Brian's Brain", "Langton's Ant"]
    )

    #parametrii lumii initiale (slidere)
    grid_size=st.slider("Dimensiunea grilei (NxN)",min_value=10, max_value=100, value=20)
    densitate=st.slider("Densitate celule vii initiale (%)",min_value=0,max_value=100, value=20)

    #setarea pentru persistenta memoriei (checkbox)

    store_history=st.checkbox("Salveaza istoricul starilor (store_history)",value=True)

    st.divider()

    st.header("🔮 Interogare Oracol (Q1-Q5)")
    st.info(f"Model selectat: {tip_automat}")

    #spatiu pt metrici, actualizarea mai jos
    oracol_q4=st.empty()
    oracol_q2 = st.empty()
    oracol_q5 = st.empty()

#initializarea memoriei globale (st.session_state)
def generare_grila_noua(size,prob_vii):
    # genereaza o grila noua folosind setarile din slidere
    np.random.seed(42)
    p_viu = prob_vii / 100.0
    p_mort = 1.0 - p_viu
    return np.random.choice([0,1], size=(size,size), p=[p_mort,p_viu]).astype(np.uint8)


#verificam daca variabilele exista deja in sesiune si daca nu, le cream din prima deschidere

if 'grid' not in st.session_state:
    st.session_state.grid=generare_grila_noua(grid_size,densitate)

if 'step' not in st.session_state:
    st.session_state.step=0


#functie ajutatoare care transforma o matrice de cifre intr o pagina colorata rgb
def grid_to_image(grid:np.ndarray, cell_size:int=15)->np.ndarray:
    """
        În C++, matricea conținea doar 0 și 1.
        Ca să o afișăm pe ecran ca o imagine clară, o convertim într-o matrice RGB:
          - 0 (Mort)  -> Alb  [255, 255, 255]
          - 1 (Viu)   -> Albastru închis [20, 50, 150]
          - 2 (Stare specială / foc) -> Roșu [220, 40, 40]
    """
    h,w=grid.shape
    #cream o matrice de pixeli(inaltime,latime,3canale de culoare)
    img=np.zeros((h,w,3), dtype=np.uint8)

    #coloram celula dupa valoarea lor
    img[grid==0]=[255,255,255] #alb
    img[grid==1]=[20,50,150] #albastru inchis
    img[grid==2]=[220,40,40] #rosu

    #np.repeat mareste fiecare celula la dimensiunea "cell_size" x "cell_size" pixeli
    img_scaled=np.repeat(np.repeat(img,cell_size,axis=0),cell_size,axis=1)
    return img_scaled


#functie temporara pt simularea unui pas (mock step)
def mock_next_step():
    st.session_state.step+=1
    #deplasam celula circular ca sa vedem schimbarea pe ecran la fiecare click
    st.session_state.grid=np.roll(st.session_state.grid,shift=1,axis=1)




st.title("CelLab - Simulator de Automate Celulare")
st.caption(f"Generația Curentă (Pasul): {st.session_state.step}")

#imp zona in 2 col: grila (stanga) si controalele (dreapta)
col_grila,col_controale=st.columns([2,1])

with col_grila:
    st.subheader("Vizualizare Generatie Curenta")
    rgb_display=grid_to_image(st.session_state.grid)
    st.image(rgb_display,caption=f"Grila {grid_size}x{grid_size} la Pasul {st.session_state.step}",clamp=True)

with col_controale:
    st.subheader("Panou de control")

    #buton de pas urmator
    if st.button("▶ Pasul Urmator (Next Step)", use_container_width=True):
        mock_next_step()
        st.rerun() #refresh pe ecran cu noile valori


    pasi_avans=st.number_input("Sari N generatii:",min_value=1,max_value=1000,value=10)
    if st.button(f"⏩ Avanseaza {pasi_avans} generatii",use_container_width=True):
        for i in range(pasi_avans):
            mock_next_step()
        st.rerun()

    #buton de resetare
    if st.button("🔄 Generare Lume Noua", use_container_width=True):
        np.random.seed(42)
        st.session_state.grid=generare_grila_noua(grid_size,densitate)
        st.session_state.step=0
        st.rerun()

    st.divider()
    st.subheader("Persistenta (Salvare/Incarcare)")

    #salvarea starii curente
    save_name=st.text_input("Nume fisier salvare:",value="salvare curenta")
    col_s,col_t=st.columns(2)
    with col_s:
        if st.button("💾 Salveaza Starea", use_container_width=True):
            pachet_date = {'grid': st.session_state.grid,
                'step': st.session_state.step,
                'spec': '{"kind": "life_like"}',
                'seed': 42}

            save_simulation(pachet_date,save_name)
            st.success(f"Salvat cu succes ca  '{save_name}' (.json + .npz)")

    with col_t:
        #incarcarea starii
        if st.button("📂 Incarca Starea", use_container_width=True):
            date_incarcate=load_simulation(save_name)
            if date_incarcate is not None:
                st.session_state.grid=date_incarcate['grid']
                st.session_state.step=date_incarcate['step']
                st.success(f"Am incarcat pasul {st.session_state.step}")
                st.rerun()
            else:
                st.error("Fisierele specificate nu au fost gasite.")

#actualizare oracol
celule_vii=int(np.count_nonzero(st.session_state.grid == 1))
oracol_q4.metric(label="Q4: Celule active (vii)", value=celule_vii)
oracol_q2.markdown("**Q2: Stare stabilă?**\nNu (mock engine)")
oracol_q5.markdown("**Q5: Ciclu detectat?**\nNu (mock engine)")