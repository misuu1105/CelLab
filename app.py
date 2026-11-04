import streamlit as st
import numpy as np
from storage import save_simulation, load_simulation
from cel_core import (
    SimulationEngine,
    AutomataOracle,
    WorldState,
    LifeLikeSpec,
    GenerationsSpec,
    LangtonSpec,
    ForestFireSpec,
)
from cel_core.init_world import place_preset, random_grid

#configurarea ferestrei din browser, prima instructiune streamlit din fisier
st.set_page_config(
    page_title="CelLab - Cellular Automata",
    layout="wide"
)

#maparea optiunilor din meniu la instante concrete dde reguli
SPEC_MAP={
    "Game of Life": LifeLikeSpec(),
    "Brian's Brain": GenerationsSpec(),
    "Langton's Ant": LangtonSpec(),
    "Forest Fire": ForestFireSpec(p_growth=0.02, p_lightning=0.001),
}


#meniul lateral cu parametri si setari
with st.sidebar:
    st.header("⚙️ Setări Simulare")

    #alegerea modelului: dropdown/selectbox
    tip_automat=st.selectbox(
        "Alege Automatul Celular:",
        list(SPEC_MAP.keys())
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


#functii auxiliare
def create_initial_engine(selected_type:str, size:int, density:int, history_flag:bool):
    #creeaza instanta de simulationengine folosind generatorii din cel_core
    chosen_spec=SPEC_MAP[selected_type]
    shape=(size,size)

    if selected_type == "Langton's Ant":
        #pt langton, plasam presetul cu o furnica in centru
        initial_state=place_preset("single_ant",shape)
    else:
        #pt restul o grila aleatoare cu densitate selectata
        grid=random_grid(shape,{1:density/100.0})
        initial_state=WorldState(grid=grid)

    return SimulationEngine(
        spec=chosen_spec,
        initial_state=initial_state,
        store_history=history_flag,
    )




#functie ajutatoare care transforma o matrice de cifre intr o pagina colorata rgb
def grid_to_image(grid:np.ndarray, agents=None, cell_size:int=15)->np.ndarray:
    """
        În C++, matricea conținea doar 0 și 1.
        Ca să o afișăm pe ecran ca o imagine clară, o convertim într-o matrice RGB:
          - 0 (Mort)  -> Alb  [255, 255, 255]
          - 1 (Viu)   -> Albastru închis [20, 50, 150]
          - 2 (Stare specială / foc) -> Roșu [220, 40, 40]

          Transformă matricea în imagine RGB.
    Dacă există agenți (ex: Langton's Ant), le evidențiază poziția cu Magenta.
    """
    h,w=grid.shape
    #cream o matrice de pixeli(inaltime,latime,3canale de culoare)
    img=np.zeros((h,w,3), dtype=np.uint8)

    #coloram celula dupa valoarea lor
    img[grid==0]=[255,255,255] #alb
    img[grid==1]=[20,50,150] #albastru inchis
    img[grid==2]=[220,40,40] #rosu


    if agents:
        for agent in agents:
            #asiguram coordonatele in interiorul grilei
            ay,ax=agent.y%h,agent.x%w
            img[ay,ax]=[255,0,255]

    #np.repeat mareste fiecare celula la dimensiunea "cell_size" x "cell_size" pixeli
    img_scaled=np.repeat(np.repeat(img,cell_size,axis=0),cell_size,axis=1)
    return img_scaled



#initializare engine in sesiune
if 'engine' not in st.session_state:
    st.session_state.engine=create_initial_engine(tip_automat,grid_size,densitate,store_history)


#preluam starea curenta din motor
current_state=st.session_state.engine.state
grid=current_state.grid
step=current_state.step




#interfata principala
st.title("CelLab - Simulator de Automate Celulare")
st.caption(f"Generația: {step} | Model:{tip_automat}")

#imp zona in 2 col: grila (stanga) si controalele (dreapta)
col_grila,col_controale=st.columns([2,1])

with col_grila:
    st.subheader("Vizualizare Univers")
    agents_list=getattr(current_state, "agents",None)
    rgb_display=grid_to_image(grid,agents=agents_list)
    st.image(rgb_display,caption=f"Grila {grid.shape[0]}x{grid.shape[1]} la Pasul {step}",clamp=True)

with col_controale:
    st.subheader("Panou de control")

    #buton de pas urmator individual
    if st.button("▶ Pasul Urmator (Next Step)", use_container_width=True):
        st.session_state.engine.step()
        st.rerun() #refresh pe ecran cu noile valori


    pasi_avans=st.number_input("Sari N generatii:",min_value=1,max_value=1000,value=10)
    if st.button(f"⏩ Avanseaza {pasi_avans} generatii",use_container_width=True):
        st.session_state.engine.run(pasi_avans)
        st.rerun()

    #buton de resetare
    if st.button("🔄 Generare Lume Noua", use_container_width=True):
        st.session_state.engine=create_initial_engine(tip_automat,grid_size,densitate,store_history)
        st.rerun()

    st.divider()
    st.subheader("Persistenta (Salvare/Incarcare)")

    #salvarea starii curente
    save_name=st.text_input("Nume fisier salvare:",value="salvare curenta")
    col_s,col_t=st.columns(2)
    with col_s:
        if st.button("💾 Salveaza Starea", use_container_width=True):
            engine=st.session_state.engine
            save_simulation(engine.state,engine.spec,save_name)
            st.success(f"Salvat cu succes ca  '{save_name}' (.json + .npz)")

    with col_t:
        #incarcarea starii
        if st.button("📂 Incarca Starea", use_container_width=True):
            date_incarcate=load_simulation(save_name)
            if date_incarcate is not None:
                loaded_state, loaded_spec= date_incarcate
                st.session_state.engine=SimulationEngine(
                    spec=loaded_spec,
                    initial_state=loaded_state,
                    store_history=store_history
                )
                st.success(f"Am incarcat pasul {loaded_state.step}")
                st.rerun()
            else:
                st.error("Fisierele specificate nu au fost gasite.")

#actualizare oracol

try:
    oracle=AutomataOracle(st.session_state.engine)

    #q4:numarare celule active
    res_q4=oracle.count_at(state=1, n=st.session_state.engine.current_step)
    active_count=getattr(res_q4,"count",res_q4)
    oracol_q4.metric(label="Q4:Celule active (starea 1",value=active_count)

    #q2:verificare stabilitate
    q2_res=oracle.stability(max_steps=100)
    stabil_text=f"Da (la pasul{q2_res.step})" if q2_res.reached else "Nu"
    oracol_q2.markdown(f"**Q2:Stabil?** {stabil_text}")

    #q5: detectie ciclu
    q5_res=oracle.cycle(max_steps=100)
    if q5_res.has_cycle:
        oracol_q5.markdown(f"**Q5: Ciclu detectat!**\nPerioada:{q5_res.period_l}, Transient: {q5_res.transient_t}")
    else:
        regime=getattr(q5_res,"regime","Niciunul")
        oracol_q5.markdown(f"**Q5: Regim:** {regime}")

except Exception as e:
    oracol_q4.write(f"Eroare Oracol: {e}")