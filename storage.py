import numpy as np
import json
import os
from cel_core import WorldState, parse_spec, BaseAutomatonSpec

def save_simulation(state:WorldState, spec:BaseAutomatonSpec, base_filename: str):
    """
        Funcția asta face un fel de 'fout'.
        Primeste un dicționar cu toate variabilele (ca un struct din C++) și un nume de fisier.
        Salvam matricea in format comprimat (.npz) si restul datelor in format text (.json).

        Salvează starea și specificația automatului:
    - Matricea (grid) merge comprimată în .npz
    - Restul stării + specificația regulilor merg în .json
    """

    state_dict=state.to_dict()

    #extragem matricea/grila din dictionar si o salvam separat in format binar comprimat
    grid=state_dict.pop('grid',state.grid)
    #salvam matricea pe disc folosind numpy
    npz_filename=f"{base_filename}.npz"
    np.savez_compressed(npz_filename, grid=grid)

    #serializam specificatia folosind direct json de la pydantic(converteste automat seturile la model_dump_json)
    if hasattr(spec, "model_dump_json"):
        spec_data=json.loads(spec.model_dump_json())
    elif hasattr(spec, "model_dump_"):
        spec_data=spec.model_dump(mode="json")
    else:
        spec_data=spec

    payload={
        "spec":spec_data,
        "state":state_dict
    }

    #functie de conversie pt orice tip necuscut din json standard
    def default_serializer(obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)


    #salvam restul metadatelor intr-un json
    json_filename=f"{base_filename}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        #json.dump transforma dictionarul python in text si il scrie in fisierul f
        json.dump(payload, f, indent=4, default=default_serializer)

    print(f"Simulare salvată cu succes în {base_filename}.json și {base_filename}.npz")


def load_simulation(base_filename: str):
    """
        Funcția asta face un fel de 'ifstream fin'.
        Citește JSON-ul și NPZ-ul și reconstruiește dicționarul.

        Încarcă fișierele și reconstruiește obiectele WorldState și BaseAutomatonSpec.
    Returnează (state, spec) sau None dacă fișierele nu există.
    """
    json_filename=f"{base_filename}.json"
    npz_filename=f"{base_filename}.npz"

    if not os.path.exists(json_filename) or not os.path.exists(npz_filename):
        print("Eroare: Fisierele nu exista!")
        return None

    #citim partea de text (step,reguli)
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    #citim matricea
    loaded_npz=np.load(npz_filename)
    grid=loaded_npz['grid']

    #reconstruim specif regulii
    spec=parse_spec(data["spec"])
    #punem matricea inapoi in dictionar
    state_dict=data["state"]
    state_dict["grid"]=grid
    state=WorldState.from_dict(state_dict)

    return state,spec