# summarizers_app.py
import math
import tkinter as tk
from tkinter import messagebox

import numpy as np

from text_rank import TextRankSummarizer
from lex_rank import LexRankSummarizer
from utils import sentenize
from graph_visualization import visualize_similarity_graph


class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Генератор заголовков")

        self.algorithm_var = tk.StringVar()
        self.algorithm_var.set("Text Rank")

        self.create_widgets()


        self.text_rank = TextRankSummarizer()
        self.lex_rank = LexRankSummarizer()

    def show_graph(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Ошибка", "Пожалуйста, введите текст.")
            return

        original_sentences = sentenize(text)
        from utils import tokenize_sentence

        tokenized = [tokenize_sentence(s) for s in original_sentences]

        graph = self.text_rank._create_graph(tokenized)

        normed = self.text_rank._norm_graph(graph)

        ranks = self.text_rank._iterate(normed)

        top_n = 3
        sorted_indices = sorted(range(len(ranks)), key=lambda i: ranks[i], reverse=True)
        summary_indices = sorted_indices[:top_n]

        visualize_similarity_graph(
            graph_matrix=normed,
            sentences=original_sentences,
            ranks=ranks,
            title="Граф",
            summary_indices=summary_indices
        )

    def add_context_menu(self, widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Копировать", command=lambda: self.copy_from_widget(widget))
        menu.add_command(label="Вставить", command=lambda: self.paste_to_widget(widget))
        menu.add_command(label="Вырезать", command=lambda: self.cut_from_widget(widget))

        def show_menu(event):
            menu.tk_popup(event.x_root, event.y_root)

        widget.bind("<Button-3>", show_menu)

    def copy_from_widget(self, widget):
        try:
            selection = widget.selection_get()
            self.root.clipboard_clear()
            self.root.clipboard_append(selection)
        except:
            pass

    def paste_to_widget(self, widget):
        try:
            widget.insert(tk.INSERT, self.root.clipboard_get())
        except:
            pass

    def cut_from_widget(self, widget):
        self.copy_from_widget(widget)
        try:
            widget.delete("sel.first", "sel.last")
        except:
            pass

    def create_widgets(self):
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(5, weight=1)
        self.root.columnconfigure(0, weight=1)

        tk.Label(self.root, text="Введите текст:").grid(row=0, column=0, sticky="w", padx=5, pady=2)

        self.text_input = tk.Text(self.root, height=10, wrap="word")
        self.text_input.grid(row=1, column=0, sticky="nsew", padx=5)

        visualize_button = tk.Button(self.root, text="Показать граф", command=self.show_graph)
        visualize_button.grid(row=2, column=0, sticky="ew", padx=5, pady=2)

        tk.Label(self.root, text="Выберите алгоритм:").grid(row=3, column=0, sticky="w", padx=5, pady=2)

        algorithm_menu = tk.OptionMenu(self.root, self.algorithm_var, "Text Rank", "Lex Rank")
        algorithm_menu.grid(row=4, column=0, sticky="ew", padx=5)

        generate_button = tk.Button(self.root, text="Сгенерировать заголовок", command=self.generate_title)
        generate_button.grid(row=5, column=0, sticky="ew", padx=5, pady=2)

        tk.Label(self.root, text="Заголовок:").grid(row=6, column=0, sticky="w", padx=5, pady=2)

        self.title_output = tk.Text(self.root, height=5, wrap="word")
        self.title_output.grid(row=7, column=0, sticky="nsew", padx=5, pady=(0, 5))

        self.root.rowconfigure(7, weight=1)

        self.add_context_menu(self.text_input)
        self.add_context_menu(self.title_output)

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