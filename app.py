import tensorflow as tf
import keras as keras  
from shiny.express import input, render, ui
import h5py

file_name = h5py.File('/Users/christofferfuglkjaer/Library/Mobile Documents/com~apple~CloudDocs/Dataproject/bin_model_syn_train.h5', 'r')
model = keras.models.load_model(file_name)



ui.page_opts(title = "cleft lib ")

with ui.sidebar():
    ui.input_numeric("Anteroposterior_1", "Anteroposterior_1", value=False, min = 0, max = 12, step = 3 )
    ui.input_numeric("Anteroposterior_2", "Anteroposterior_2", value=True)
    ui.input_numeric("Vertical_1", "Vertical_1", value=False)
    ui.input_numeric("Vertical_2", "Vertical_2", value="Hello, world!")
    ui.input_numeric("Goslon_Score_A", "Goslon_Score_A", value=True)
    ui.input_numeric("Total_Row_Score_A", "Total_Row_Score_A	", value=False)



def server(input, output, session):
    @output
    def main():
       y = model.summay()
       return y 




# print("Anteroposterior_1", "Anteroposterior_1")
# print("Anteroposterior_2", "Anteroposterior_2")
# print("Vertical_1", "Vertical_1")
# print("Vertical_2", "Vertical_2")
# print("Goslon_Score_A", "Goslon_Score_A")
# print("Total_Row_Score_A", "Total_Row_Score_A	")

# print("Anteroposterior_1", "Anteroposterior_1")
 