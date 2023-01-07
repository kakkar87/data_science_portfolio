import numpy as np
import pickle
import streamlit as st

# loadig the saved model
loaded_model = pickle.load(open('C:/Users/arpit/Juypter/Diabetes/trained_model_svm.sav', 'rb'))

def diabetes_prediction(input_data):

    input_data = [5,166,72,19,175,25.8,0.587,51]

    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "Person is Non-Diabetic"
    else:
        return 'Person is Diabetic'

def main():

    # giving a title
    st.title('Diabetes Prediction Web App')

    # giving the input date from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')

    # code for prediction
    diagnosis = ''

    # creating a button for prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
    




