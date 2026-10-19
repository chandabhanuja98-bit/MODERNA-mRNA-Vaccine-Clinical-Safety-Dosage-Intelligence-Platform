import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rag_index import query_guidelines

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
from schemas import ClinicalSafetyReport
import time
from pathlib import Path
import json

load_dotenv()

SYSTEM_PROMPT = """
You are generating a structured report for a student engineering
project. This is not a real clinical decision-making system.

Follow these rules strictly:

1. Use only the provided ML risk output and retrieved guideline
   excerpts as factual evidence.

2. Never recommend a dose that contradicts the retrieved
   guideline text.

3. Never invent clinical facts, statistics, safety warnings,
   or guidelines that are absent from the retrieved excerpts.

4. If the retrieved excerpts do not provide sufficient
   information, set Eligible_For_Vaccine to false and explain
   the uncertainty in Reasoning_Steps.

5. Include exact relevant excerpts in FDA_Safety_Warnings_Cited.
   Do not fabricate quotations.

6. The report must always include this disclaimer:
   "Student engineering project output — not real clinical guidance."

7. The required output fields are:
   Eligible_For_Vaccine,
   Recommended_Dose_mg,
   Reasoning_Steps,
   FDA_Safety_Warnings_Cited,
   Disclaimer.

8. The reference guidelines may express doses in micrograms.
   Convert units to milligrams only using the exact conversion:
   1000 micrograms = 1 milligram.
   Never invent a dose when the evidence is insufficient.

Return a response that conforms to the required report schema.
"""


model_path = "outputs/risk_model.joblib"

numeric_columns = ['Age','Biomarker_Level','Vaccine_Dose_mcg']
categorical_columns = 'Prior_Reaction_History'

feature_columns = numeric_columns + [categorical_columns,'Pre_Existing_Conditions'] 


clean_data_path = "data/patient_records_clean.csv"

def load_risk_model(model_path):
    model = joblib.load(model_path)
    print("Loaded object type:", type(model))
    return model



def sample_patient(clean_data_path):
    read_data = pd.read_csv(clean_data_path)
    patient = read_data.iloc[[0]]
    return patient

def prepare_patient_features(patient):
    # Read the same cleaned dataset used during Week 1 training
    training_data = pd.read_csv(clean_data_path)

    # Recreate the fitted preprocessing objects
    scaler = StandardScaler()
    scaler.fit(training_data[numeric_columns])

    encoder = LabelEncoder()
    encoder.fit(training_data[categorical_columns])

    # Transform the sample patient
    prepared_patient = patient.copy()

    prepared_patient[numeric_columns] = scaler.transform(
        prepared_patient[numeric_columns]
    )

    prepared_patient[categorical_columns] = encoder.transform(
        prepared_patient[categorical_columns]
    )

    # Return only the model's input features in the correct order
    return prepared_patient[feature_columns]

def laod_rag_index():
    model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="moderna_guidelines",
        embedding_function=model,
        persist_directory="outputs/chroma_store"
    )
    return vector_store


def generate_report(
    risk_label : str,
    risk_probability : float,
    guidelines_chunks : list[str],
    patient_summary : str
) -> ClinicalSafetyReport :

    parser = PydanticOutputParser(pydantic_object=ClinicalSafetyReport)
    prompt = ChatPromptTemplate.from_messages([
        ("system",SYSTEM_PROMPT+"\n\n {format_instructions}"),
        ("user",
            """
            Patient summary:
            {patient_summary}

            Predicted reaction-risk category:
            {risk_label}

            Estimated probability of high risk:
            {risk_probability}

            Retrieved guideline excerpts:
            {guidelines}

            {retry_feedback}

            Generate the structured report using only the
            supplied evidence. Do not guess if evidence is
            insufficient.
            """,),

    ])
    llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash",temperature = 0)

    inputs = {
        "patient_summary" : patient_summary,
        "risk_label" : risk_label,
        "risk_probability" : f"{risk_probability:.2%}",
        "guidelines" : guidelines_chunks,
        "format_instructions" : parser.get_format_instructions(),
        "retry_feedback" : "",
    }
    for attempt in range(2) :
        messages = prompt.format_messages(**inputs)
        response = llm.invoke(messages)
        try :
            content = response.content

            if isinstance(content, list):
                text_parts = [
                    item["text"]
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ]
                content = "\n".join(text_parts)

            if not isinstance(content, str):
                raise ValueError(
                    f"Unexpected Gemini response content type: {type(content)}"
                )

            report = parser.parse(content)
            return report
        except(OutputParserException,ValidationError) as error :
            if attempt == 1 : 
                raise ValueError("report validation failed after one retry ") from error
            inputs["retry_feedback"]=(
                "your previous response failed validation "
                f"validation error : {error}"
                "Correct the response and return the valid output"
                "Maching the required schema"
            )
    raise RuntimeError("report generation didn't complete")    
            



if __name__ == "__main__" :
    model = load_risk_model(model_path)
    print("-------------------- The model is loaded successfully -----------------------------------")

    patient = sample_patient(clean_data_path)
    print(" The sample patient : ")
    print(patient)

    print(" the prepared patient : ")
    new_patient = prepare_patient_features(patient)
    print(new_patient)

    prediction = model.predict(new_patient)[0]
    class_index = list(model.classes_).index(1)
    risk_probability = model.predict_proba(new_patient)[0][class_index]
    risk_label = "High" if prediction == 1 else "Low"
    print(" The risk prediction is ")
    print(" predicted risk : ",risk_label)
    print(" Probability of risk : ",risk_probability)

    vector_store = laod_rag_index()

    row = patient.iloc[0]
    question = (
        f"Find relevant safety warnings, contraindications, and "
        f"dosing guidance for a patient aged {row['Age']} years, "
        f"with a candidate vaccine dose of "
        f"{row['Vaccine_Dose_mcg']} micrograms, prior reaction "
        f"history {row['Prior_Reaction_History']}, and "
        f"{row['Pre_Existing_Conditions']} pre-existing conditions."
    )
    guideline_chunks , latency = query_guidelines(vector_store,question,k=3)
    print("---------------------------- retrive safety guidelines ------------------------------------")
    for index , chunk in enumerate(guideline_chunks,start=1) :
        print(f"\n guideline {index} : \n {chunk}")
    print(f"\n retrive latency : {latency:.3f} seconds ")

    patient_summary = (
        f"Age: {row['Age']}; "
        f"Biomarker level: {row['Biomarker_Level']}; "
        f"Candidate dose: {row['Vaccine_Dose_mcg']} mcg; "
        f"Prior reaction history: {row['Prior_Reaction_History']}; "
        f"Pre-existing conditions: "
        f"{row['Pre_Existing_Conditions']}"
    )
    print("----------------------------------Generating report with Gemini----------------------------------")
    start_time = time.perf_counter()
    report = generate_report(
        risk_label=risk_label,
        risk_probability= float(risk_probability),
        guidelines_chunks=guideline_chunks,
        patient_summary=patient_summary
    )
    print("----------------------------------- Clinical safety report ---------------------------------------")
    print(report)

    llm_report_latency = time.perf_counter()-start_time
    latency_path = Path("outputs/reports/latencies.json")
    latency_path.parent.mkdir(parents=True,exist_ok=True)
    latency_data = {
        "rag_query_seconds":float(latency),
        "Gemini_report_seconds" : float(llm_report_latency)     
    }
    with latency_path.open("w",encoding="utf-8")as file : 
        json.dump(latency_data,file,indent=2)
    
    print(f"Gemini latency : {llm_report_latency : .3f} seconds")
    print(report.model_dump_json(indent=2))



    


    


    

    






