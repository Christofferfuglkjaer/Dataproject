from shiny.express import input, render, ui

ui.input_numeric("a1", "a1", value="Hello, world!")
ui.input_numeric("a2", "a2", value=True)
ui.input_numeric("italic", "Italic", value=False)

ui.input_numeric("message", "Message", value="Hello, world!")
ui.input_numeric("bold", "Bold", value=True)
ui.input_numeric("italic", "Italic", value=False)
ui.input_numeric("a1", "a1", value="Hello, world!")
ui.input_numeric("a2", "a2", value=True)
ui.input_numeric("italic", "Italic", value=False)

ui.input_numeric("message", "Message", value="Hello, world!")
ui.input_numeric("bold", "Bold", value=True)
ui.input_numeric("italic", "Italic", value=False)
ui.input_numeric("italic", "Italic", value=False)



@render.ui
def result():
    x = input.message()
    if "Bold" in input.styles():
        x = ui.strong(x)
    if "Italic" in input.styles():
        x = ui.em(x)
    return x