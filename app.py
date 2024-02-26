from shiny.express import input, render, ui

ui.page_opts(title="Clef project")


with ui.sidebar():
    ui.input_numeric("Anteroposterior_1", "Anteroposterior_1", value="Hello, world!")
    ui.input_numeric("Anteroposterior_2", "Anteroposterior_2", value=True)
    ui.input_numeric("Vertical_1", "Vertical_1", value=False)

    ui.input_numeric("Vertical_2", "Vertical_2", value="Hello, world!")
    ui.input_numeric("Goslon_Score_A", "Goslon_Score_A", value=True)
    ui.input_numeric("Total_Row_Score_A", "Total_Row_Score_A	", value=False)



@render.ui
def result():
    x = input.message()
    if "Anteroposterior_1" is 2:
        print(2)
    if "Italic" in input.styles():
        x = ui.em(x)
    return x



# print("Anteroposterior_1", "Anteroposterior_1")
# print("Anteroposterior_2", "Anteroposterior_2")
# print("Vertical_1", "Vertical_1")
# print("Vertical_2", "Vertical_2")
# print("Goslon_Score_A", "Goslon_Score_A")
# print("Total_Row_Score_A", "Total_Row_Score_A	")

# print("Anteroposterior_1", "Anteroposterior_1")
def main():
    pass