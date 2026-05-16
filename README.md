# Difference Schemes Analyser

Анализ разностных схем для волновых уравнений с краевыми условиями Дирихле

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0+-orange.svg)](https://matplotlib.org/)

## О проекте

Программа предназначена для численного решения волновых уравнений с краевыми условиями Дирихле. 
Аппроксимация начального условия выполняется со **вторым порядком точности** с использованием разложения в ряд Тейлора в окрестности точки ноль по τ.

### Особенности

- Два различных численных метода решения
- Визуализация результатов
- Анализ устойчивости схем
- Автоматическое исследование сходимости

### Установка зависимостей

```bash
pip install numpy matplotlib pandas