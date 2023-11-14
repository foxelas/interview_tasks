import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import scrolledtext
import threading

from main import build_and_evaluate_model_target_data


class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        self.text_space.insert('end', string)
        self.text_space.see('end')

    def flush(self):
        pass


class Window:
    def __init__(self, master):
        self.master = master

        self.notebook = ttk.Notebook(self.master)
        self.frame1 = ttk.Frame(self.notebook, name="settingstab", width=50)
        self.frame1.grid(row=0, column=0, sticky="nsew")

        label_experiment_name = ttk.Label(self.frame1, text="Experiment Name")
        label_experiment_name.pack(padx=5, pady=5)
        self.textbox_experiment_name = tk.Text(self.frame1, height=1)
        self.textbox_experiment_name.pack(padx=2, pady=2)
        self.textbox_experiment_name.insert("1.0", "Test")

        label_location_id = ttk.Label(self.frame1, text="Location ID")
        label_location_id.pack(padx=5, pady=5)
        self.textbox_location_id = tk.Text(self.frame1, height=1)
        self.textbox_location_id.pack(padx=2, pady=2)
        self.textbox_location_id.insert("1.0", "116.0")

        label_value_type_id = ttk.Label(self.frame1, text="Value Type ID")
        label_value_type_id.pack(padx=5, pady=5)
        self.textbox_value_type_id = tk.Text(self.frame1, height=1)
        self.textbox_value_type_id.pack(padx=2, pady=2)
        self.textbox_value_type_id.insert("1.0", "11")

        label_activation = ttk.Label(self.frame1, text="Activation")
        label_activation.pack(padx=5, pady=5)
        self.activation_function = tk.StringVar()
        self.combobox_activation = ttk.Combobox(self.frame1, textvariable=self.activation_function, width=90)
        self.combobox_activation['values'] = ('relu', 'sigmoid')
        self.combobox_activation['state'] = 'normal'
        self.combobox_activation.current(0)
        self.combobox_activation.pack(padx=2, pady=2)

        label_layers_number = ttk.Label(self.frame1, text="Layers Number")
        label_layers_number.pack(padx=5, pady=5)
        self.textbox_layers_number = tk.Text(self.frame1, height=1)
        self.textbox_layers_number.pack(padx=2, pady=2)
        self.textbox_layers_number.insert("1.0", "3")

        label_neurons_number = ttk.Label(self.frame1, text="Neurons Number")
        label_neurons_number.pack(padx=5, pady=5)
        self.textbox_neurons_number = tk.Text(self.frame1, height=1)
        self.textbox_neurons_number.pack(padx=2, pady=2)
        self.textbox_neurons_number.insert("1.0", "32, 16, 8, 4")

        label_loss = ttk.Label(self.frame1, text="Loss")
        label_loss.pack(padx=5, pady=5)
        self.loss_function = tk.StringVar()
        self.combobox_loss = ttk.Combobox(self.frame1, textvariable=self.loss_function, width=90)
        self.combobox_loss['values'] = ('mean_squared_error', 'mean_absolute_error')
        self.combobox_loss['state'] = 'normal'
        self.combobox_loss.current(0)
        self.combobox_loss.pack(padx=2, pady=2)

        label_optimizer = ttk.Label(self.frame1, text="Optimizer")
        label_optimizer.pack(padx=5, pady=5)
        self.optimizer = tk.StringVar()
        self.combobox_optimizer = ttk.Combobox(self.frame1, textvariable=self.optimizer, width=90)
        self.combobox_optimizer['values'] = ('SGD', 'Adam')
        self.combobox_optimizer['state'] = 'normal'
        self.combobox_optimizer.current(0)
        self.combobox_optimizer.pack(padx=2, pady=2)

        label_epochs = ttk.Label(self.frame1, text="Epochs")
        label_epochs.pack(padx=5, pady=5)
        self.textbox_epochs = tk.Text(self.frame1, height=1)
        self.textbox_epochs.pack(padx=2, pady=2)
        self.textbox_epochs.insert("1.0", "200")

        label_batch_size = ttk.Label(self.frame1, text="Batch Size")
        label_batch_size.pack(padx=5, pady=5)
        self.textbox_batch_size = tk.Text(self.frame1, height=1)
        self.textbox_batch_size.pack(padx=2, pady=2)
        self.textbox_batch_size.insert("1.0", "32")

        label_window_size = ttk.Label(self.frame1, text="Window Size")
        label_window_size.pack(padx=5, pady=5)
        self.textbox_window_size = tk.Text(self.frame1, height=1)
        self.textbox_window_size.pack(padx=2, pady=2)
        self.textbox_window_size.insert("1.0", "20")

        label_split_percentage = ttk.Label(self.frame1, text="Split Percentage")
        label_split_percentage.pack(padx=5, pady=5)
        self.textbox_split_percentage = tk.Text(self.frame1, height=1)
        self.textbox_split_percentage.pack(padx=2, pady=2)
        self.textbox_split_percentage.insert("1.0", "0.2")

        run_button = ttk.Button(self.frame1, text='Run')
        run_button['command'] = self.click_run
        run_button.pack(pady=5, padx=5)
        self.frame1.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.frame1, text="Settings")

        ############# Frame 2 #############
        self.frame2 = ttk.Frame(self.notebook, name="resultstab")
        self.frame2.grid(row=0, column=0, sticky="nsew")

        self.textbox_result = scrolledtext.ScrolledText(self.frame2, undo=True, width=200)
        self.textbox_result.grid(column=0, row=0, sticky='NSWE', padx=5, pady=5)
        self.textbox_result.pack(fill=tk.BOTH, expand=True)

        self.frame2.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.frame2, text="Results")
        self.redirect_print_to_tab()

        self.notebook.pack(padx=5, pady=5, expand=True)

    def redirect_print_to_tab(self):
        sys.stdout = StdoutRedirector(self.textbox_result)

    def change_to_result(self):
        self.notebook.select(1)

    def run_main(self):
        experiment_name = self.textbox_experiment_name.get("1.0", "end-1c")
        location_id = float(self.textbox_location_id.get("1.0", "end-1c"))
        value_type_id = int(self.textbox_value_type_id.get("1.0", "end-1c"))
        activation_function = self.activation_function.get()
        layers_number = int(self.textbox_layers_number.get("1.0", "end-1c"))
        neurons_number = [int(x) for x in self.textbox_neurons_number.get("1.0", "end-1c").split(',')]
        loss_function = self.loss_function.get()
        optimizer = self.optimizer.get()
        epochs = int(self.textbox_epochs.get("1.0", "end-1c"))
        batch_size = int(self.textbox_batch_size.get("1.0", "end-1c"))
        window_size = int(self.textbox_window_size.get("1.0", "end-1c"))
        split_percentage = float(self.textbox_split_percentage.get("1.0", "end-1c"))

        build_and_evaluate_model_target_data(experiment_name, location_id, value_type_id, window_size,
                                             activation_function, layers_number, neurons_number, loss_function,
                                             optimizer, epochs, batch_size, window_size, split_percentage)

    def click_run(self):
        self.change_to_result()
        threading.Thread(target=self.run_main).start()


root = tk.Tk()
window = Window(root)
root.mainloop()
