import pandas as pd

def get_ai_diet_exercise(disease, symptoms):

    if disease is None:
        return {"diet": "No specific diet found", "exercise": "No specific exercise found"}

    diet_df= pd.read_csv("Dataset/diets.csv")

    excercise_df = pd.read_csv("Dataset/Exercise.csv")
    match = diet_df[diet_df["Disease"].str.lower() == disease.lower()]
    match_E = excercise_df[excercise_df["Disease"].str.lower() == disease.lower()]
    if not match.empty:
        diet = match["Diet"].values[0]
        exercise = match_E["Exercise"].values[0]
        return {"diet": diet, "exercise": exercise}
    return {"diet": "No specific diet found", "exercise": "No specific exercise found"}

def get_precautions(disease):

    if disease is None:
        return ["No specific precautions found"]
    
    precautions_df = pd.read_csv("Dataset/precautions_df.csv")
    match = precautions_df[precautions_df["Disease"].str.lower() == disease.lower()]
    if not match.empty:
        Precaution_1 = match["Precaution_1"].values[0]
        Precaution_2 = match["Precaution_2"].values[0]
        Precaution_3 = match["Precaution_3"].values[0]
        Precaution_4 = match["Precaution_4"].values[0]
        return [Precaution_1, Precaution_2, Precaution_3, Precaution_4]
    return ["No specific precautions found"]