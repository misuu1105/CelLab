import numpy as np
import json
import os

def save_simulation(world_state_dict:dict, base_filename: str):
    """
        Funcția asta face un fel de 'fout'.
        Primeste un dicționar cu toate variabilele (ca un struct din C++) și un nume de fisier.
        Salvam matricea in format comprimat (.npz) si restul datelor in format text (.json).
    """
    #extragem matricea din dictionar
    grid=world_state_dict.pop('grid')
    #salvam matricea pe disc folosind numpy
    npz_filename=base_filename+'.npz'
    np.savez_compressed(npz_filename, grid=grid)
    #salvam restul datelor intr-un json
    json_filename=base_filename+'.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        #json.dump transforma dictionarul python in text si il scrie in fisierul f
        json.dump(world_state_dict, f, indent=4)

    print(f"Am salvat simularea in {npz_filename} si {json_filename}")

    #punem grila/matricea inapoi in dictionar in caz de
    world_state_dict['grid']=grid


def load_simulation(base_filename: str)->dict:
    """
        Funcția asta face un fel de 'ifstream fin'.
        Citește JSON-ul și NPZ-ul și reconstruiește dicționarul.
    """
    json_filename=base_filename+'.json'
    npz_filename=base_filename+'.npz'

    if not os.path.exists(json_filename) or not os.path.exists(npz_filename):
        print("Eroare: Fisierele nu exista!")
        return None

    #citim partea de text (step,reguli)
    with open(json_filename, 'r', encoding='utf-8') as f:
        world_state_dict = json.load(f)

    #citim matricea
    loaded_npz=np.load(npz_filename)
    grid=loaded_npz['grid']
    #punem matricea inapoi in dictionar
    world_state_dict['grid']=grid

    print(f"Am incarcat simularea de la pasul:{world_state_dict.get('step')}")

    return world_state_dict


if __name__=='__main__':
    #facem o matrice 10x10 de proba
    test_grid=np.zeros((10,10),dtype=np.uint8)
    test_grid[5][5]=1

    test_world={
        'grid':test_grid,
        'step':42,
        'spec':'{"kind","life_like","birth":[3],"survive":[2,3]}',
        'seed':12345
    }

    # Salvăm
    print("Testam Salvarea...")
    save_simulation(test_world, "test_save")

    # Încărcăm
    print("\nTestam Incarcarea...")
    loaded_world = load_simulation("test_save")

    # Afisam ce am incarcat
    print("Matricea incarcata (valoarea de la 5,5 trebuie sa fie 1):", loaded_world['grid'][5][5])