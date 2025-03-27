# main.py
import tkinter as tk
from tkinter import messagebox
from text_rank import TextRankSummarizer
from lex_rank import LexRankSummarizer
from utils import sentenize

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Генератор заголовков")

        self.algorithm_var = tk.StringVar()
        self.algorithm_var.set("Text Rank")

        self.create_widgets()

        self.text_rank = TextRankSummarizer()
        self.lex_rank = LexRankSummarizer()

    def create_widgets(self):

        tk.Label(self.root, text="Введите текст:").pack()
        self.text_input = tk.Text(self.root, height=10, width=50)
        self.text_input.pack()

        tk.Label(self.root, text="Выберите алгоритм:").pack()
        algorithm_menu = tk.OptionMenu(self.root, self.algorithm_var, "Text Rank", "Lex Rank")
        algorithm_menu.pack()

        generate_button = tk.Button(self.root, text="Сгенерировать заголовок", command=self.generate_title)
        generate_button.pack()

        tk.Label(self.root, text="Заголовок:").pack()
        self.title_output = tk.Text(self.root, height=5, width=50)
        self.title_output.pack()

    def generate_title(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Ошибка", "Пожалуйста, введите текст.")
            return

        algorithm = self.algorithm_var.get()
        sentences = sentenize(text)

        if algorithm == "Text Rank":
            summary = self.text_rank(text, target_sentences_count=1)
        elif algorithm == "Lex Rank":
            summary = self.lex_rank(sentences, sentence_count=1)
            summary = " ".join(summary)
        else:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите алгоритм.")
            return

        self.title_output.delete("1.0", tk.END)
        self.title_output.insert(tk.END, summary)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()