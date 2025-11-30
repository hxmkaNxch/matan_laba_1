'''
лабораторная работа №1
вохмяков данила никитич (504499)
вариант: 4 g(x_{n+1}) = r * x_n * (1 - x_n) * (2 - x_n)
'''

import numpy as np
import matplotlib.pyplot as plt 
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

#---основные функции---

# классическое логистическое отображение
def logistic_map(x: float, r: float) -> float:
    return r * x * (1 - x)

# отображение варианта 4
def variant_map(x: float, r: float) -> float:
    return r * x * (1 - x) * (2 - x)

# производная логистического отображения
def derivative_logistic(x: float, r: float) -> float:
    return r * (1 - 2 * x)

# производная отображения варианта 4
def derivative_variant(x: float, r: float) -> float:
    return r * (2 - 6*x + 3*x**2)

#---easy level---

# анализ неподвижных точек для варианта 4
def analyze_fixed_points(r: float) -> List[float]:
    points = [0.0]  # x = 0 всегда неподвижная точка

    if r == 0:
        return points
    
    # решаем уравнение: x = r*x*(1-x)*(2-x)
    # упрощаем: 1 = r*(1-x)*(2-x) для x ≠ 0
    # получаем: x² - 3x + (2 - 1/r) = 0
    a, b, c = 1, -3, (2 - 1/r)
    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        x1 = (-b + np.sqrt(discriminant)) / (2 * a)
        x2 = (-b - np.sqrt(discriminant)) / (2 * a)

        if 0 <= x1 <= 1:
            points.append(x1)
        if 0 <= x2 <= 1:
            points.append(x2)
    
    return points

# неподвижные точки логистического отображения
def find_logistic_fixed_points(r: float) -> List[float]:
    points = [0.0]
    if r > 1:
        x_star = 1 - 1/r
        if 0 <= x_star <= 1:
            points.append(x_star)
    return points

# сравнение логистического отображения и варианта 4
def plot_comparison():
    x = np.linspace(0, 1, 1000)
    r_values = [0.5, 1.0, 1.5, 2.0, 2.5]

    plt.figure(figsize=(15, 10))

    # 1. логистическое отображение
    plt.subplot(2, 3, 1)
    for r in r_values:
        y = logistic_map(x, r)
        plt.plot(x, y, label=f'r = {r}', linewidth=2)
    plt.plot(x, x, 'k--', alpha=0.5)
    plt.title('логистическое отображение')
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. вариант 4
    plt.subplot(2, 3, 2)
    for r in r_values:
        y = variant_map(x, r)
        plt.plot(x, y, label=f'r = {r}', linewidth=2)
    plt.plot(x, x, 'k--', alpha=0.5)
    plt.title('$rx_n(1 - x_n)(2-x_n)$')
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. сравнение при r = 2
    plt.subplot(2, 3, 3)
    r_compare = 2.0
    y_logistic = logistic_map(x, r_compare)
    y_variant = variant_map(x, r_compare)
    plt.plot(x, y_logistic, 'b-', label='логистическое', linewidth=2)
    plt.plot(x, y_variant, 'r-', label='мой вариант', linewidth=2)
    plt.plot(x, x, 'k--', alpha=0.5)
    plt.title(f'сравнение при r = {r_compare}')
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. количество неподвижных точек
    plt.subplot(2, 3, 4)
    r_range = np.linspace(0.1, 3, 100)
    fixed_points_count = [len(analyze_fixed_points(r)) for r in r_range]
    plt.plot(r_range, fixed_points_count, 'g-', linewidth=2)
    plt.title('количество неподвижных точек\n(вариант 4)')
    plt.xlabel('r')
    plt.ylabel('количество точек')
    plt.grid(True, alpha=0.3)

    # 5. максимальные значения
    plt.subplot(2, 3, 5)
    max_logistic = [np.max(logistic_map(x, r)) for r in r_range]
    max_variant = [np.max(variant_map(x, r)) for r in r_range]
    plt.plot(r_range, max_logistic, 'b-', label='логистическое', linewidth=2)
    plt.plot(r_range, max_variant, 'r-', label='мой вариант', linewidth=2)
    plt.title('максимальные значения')
    plt.xlabel('r')
    plt.ylabel('max $x_{n+1}$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. положение максимума
    plt.subplot(2, 3, 6)
    # для варианта 4: максимум при производной = 0
    # g'(x) = r*(2 - 6x + 3x²) = 0 ⇒ 3x² - 6x + 2 = 0
    x_max_var = (6 - np.sqrt(36 - 24)) / 6  # ≈ 0.4226
    x_max_logistic = 0.5

    plt.bar(['логистическое', 'мой вариант'],
            [x_max_logistic, x_max_var],
            color=['blue', 'red'], alpha=0.7)
    plt.title('положение максимума')
    plt.ylabel('x координата максимума')

    plt.tight_layout()
    plt.savefig('easy_level_results.png', dpi=300, bbox_inches='tight')
    plt.show()

#---normal level---

# построение графиков последовательностей для разных r
def plot_sequences():
    r_values = [0.3, 0.7, 1.2, 1.8, 2.3]
    n_iter = 50
    x0 = 0.3
    
    plt.figure(figsize=(12, 8))
    
    for i, r in enumerate(r_values):
        # генерация последовательности для варианта 4
        x_sequence = [x0]
        for n in range(n_iter):
            x_next = variant_map(x_sequence[-1], r)
            x_sequence.append(x_next)
        
        # построение графика
        plt.subplot(2, 3, i+1)
        plt.plot(range(n_iter+1), x_sequence, 'b-', linewidth=2, marker='o', markersize=3)
        plt.title(f'r = {r}')
        plt.xlabel('n')
        plt.ylabel('$x_n$')
        plt.grid(True, alpha=0.3)
        
        # добавляем неподвижные точки
        fixed_points = analyze_fixed_points(r)
        for fp in fixed_points:
            if fp > 0:
                plt.axhline(y=fp, color='red', linestyle='--', alpha=0.7, label=f'x* = {fp:.3f}')
        
        if i == 0 and any(fp > 0 for fp in fixed_points):
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('normal_level_sequences.png', dpi=300, bbox_inches='tight')
    plt.show()

# анализ монотонности подпоследовательностей
def monotonicity_analysis():
    print('---анализ монотонности---')
    
    # для логистического отображения при r ∈ (2;3)
    r = 2.5
    x_star = 1 - 1/r
    
    x0 = 0.8  # > x* = 0.6
    n_iter = 10
    
    x_sequence = [x0]
    for n in range(n_iter):
        x_next = logistic_map(x_sequence[-1], r)
        x_sequence.append(x_next)
    
    print(f'r = {r}, x* = {x_star:.4f}')
    print(f'x₀ = {x0:.4f} (должен быть > x*)')
    
    even_seq = x_sequence[::2]  # x₀, x₂, x₄, ...
    odd_seq = x_sequence[1::2]  # x₁, x₃, x₅, ...
    
    print(f'вся последовательность: {[f"{x:.4f}" for x in x_sequence]}')
    print(f'чётные: {[f"{x:.4f}" for x in even_seq]}')
    print(f'нечётные: {[f"{x:.4f}" for x in odd_seq]}')
    
    # проверяем условие
    even_condition = all(x > x_star for x in even_seq[:3])
    odd_condition = all(x < x_star for x in odd_seq[:3])
    
    print(f'условие x₂ₙ > x* выполняется: {even_condition}')
    print(f'условие x₂ₙ₊₁ < x* выполняется: {odd_condition}')
    
    # проверка монотонности
    even_mono = all(even_seq[i] >= even_seq[i+1] for i in range(len(even_seq)-1))
    odd_mono = all(odd_seq[i] <= odd_seq[i+1] for i in range(len(odd_seq)-1))
    
    print(f'чётная подпоследовательность монотонно убывает: {even_mono}')
    print(f'нечётная подпоследовательность монотонно возрастает: {odd_mono}')

# анализ сходимости к 0 для варианта 4
def convergence_analysis():
    print('---анализ сходимости к 0---')
    
    print("условие сходимости к 0: |g'(0)| < 1")
    print("g'(x) = r*[(1-x)(2-x) - x(2-x) - x(1-x)]")
    print("g'(0) = r*(1*2) = 2r")
    print('|2r| < 1 ⇒ r < 0.5')
    print('диапазон: r ∈ [0, 0.5)')
    
    # экспериментальная проверка
    print('\nэкспериментальная проверка:')
    test_r = [0.2, 0.4, 0.6, 0.8]
    x0 = 0.3
    n_iter = 20
    
    for r in test_r:
        x_sequence = [x0]
        for n in range(n_iter):
            x_next = variant_map(x_sequence[-1], r)
            x_sequence.append(x_next)
        
        final_value = x_sequence[-1]
        converges_to_zero = final_value < 0.01
        print(f'r = {r}: x_{n_iter} = {final_value:.6f}, сходится к 0: {converges_to_zero}')

#---hard level---

# построение лестницы ламерея
def plot_lameyre_staircase(r: float, x0: float, n_iter: int = 20, use_variant: bool = False):
    # выбираем отображение
    map_func = variant_map if use_variant else logistic_map
    map_name = "вариант 4" if use_variant else "логистическое"
    
    # генерируем последовательность
    x_sequence = [x0]
    for n in range(n_iter):
        x_next = map_func(x_sequence[-1], r)
        x_sequence.append(x_next)

    # строим лестницу
    x_plot = np.linspace(0, 1, 1000)
    y_plot = map_func(x_plot, r)

    plt.figure(figsize=(10, 8))

    # кривая отображения и линия y = x
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'{map_name} (r = {r})')
    plt.plot(x_plot, x_plot, 'k--', alpha=0.7, label='y = x')

    # строим лестницу
    for i in range(min(n_iter, len(x_sequence) - 1)):
        # вертикальная линия: (x_i, x_i) -> (x_i, x_{i+1})
        plt.plot([x_sequence[i], x_sequence[i]],
                [x_sequence[i], x_sequence[i + 1]], 'r-', linewidth=1)
        
         # горизонтальная линия: (x_i, x_{i+1}) -> (x_{i+1}, x_{i+1})
        plt.plot([x_sequence[i], x_sequence[i+1]], 
                [x_sequence[i+1], x_sequence[i+1]], 'r-', linewidth=1)
    
    # точки последовательности
    plt.plot(x_sequence[:-1], x_sequence[1:], 'ro', markersize=4, label='итерации')
    
    title = f'лестница ламерея ({map_name}, r={r}, x₀={x0})'
    plt.title(title)
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    filename = f'lameyre_{"variant" if use_variant else "logistic"}_r{r}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# анализ циклов логистического отображения
def analyze_cycles():
    print('\n---анализ циклов логистического отображения---')
    
    r_infinity = 3.5699456
    print(f'r∞ = {r_infinity:.6f} (точка накопления)')
    print('при r ∈ (3; r∞) происходят бифуркации удвоения периода')
    print('длины циклов: 2 → 4 → 8 → 16 → ... → ∞')
    print('ограничение: m = 2^k (степени двойки)')

    # демонстрация циклов для разных r
    r_values = [3.2, 3.5, 3.55, 3.569]
    n_iter = 1000
    burn_in = 500  # отбрасываем начальные итерации

    plt.figure(figsize=(12, 8))

    for i, r in enumerate(r_values):
        # генерируем длинную последовательность
        x_sequence = [0.3]
        for n in range(n_iter):
            x_next = logistic_map(x_sequence[-1], r)
            x_sequence.append(x_next)

        # берем только установившиеся значения 
        steady_state = x_sequence[burn_in:]

        # находим уникальные значения
        unique_values = np.unique(np.round(steady_state, 6))
        cycle_length = len(unique_values)

        plt.subplot(2, 2, i+1)
        plt.plot(range(len(steady_state)), steady_state, 'b.', markersize=2, alpha=0.7)
        plt.title(f'r = {r}, цикл порядка {cycle_length}')
        plt.xlabel('n (после установления)')
        plt.ylabel('$x_n$')
        plt.grid(True, alpha=0.3)
        
        print(f'r = {r}: цикл порядка {cycle_length}')
    
    plt.tight_layout()
    plt.savefig('cycle_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# анализ циклов для варианта 4
def variant_cycle_analysis():
    print('\n---анализ циклов для варианта 4---')
    
    r_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    n_iter = 2000
    burn_in = 1000
    
    cycle_lengths = []
    
    for r in r_values:
        # генерируем последовательность
        x_sequence = [0.3]
        for n in range(n_iter):
            x_next = variant_map(x_sequence[-1], r)
            x_sequence.append(x_next)
        
        # анализируем установившийся режим
        steady_state = x_sequence[burn_in:]
        unique_values = np.unique(np.round(steady_state, 8))
        cycle_length = len(unique_values)
        cycle_lengths.append(cycle_length)
        
        # дополнительная проверка
        if cycle_length > 50:
            cycle_type = "хаос"
        elif cycle_length == 1:
            cycle_type = "неподвижная точка"
        else:
            cycle_type = f"цикл порядка {cycle_length}"
            
        print(f'r = {r}: {cycle_type}')
    
    # график зависимости длины цикла от r
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, cycle_lengths, 'bo-', linewidth=2, markersize=6)
    plt.title('зависимость длины цикла от r (вариант 4)')
    plt.xlabel('r')
    plt.ylabel('порядок цикла')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('variant_cycles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('\n---сравнение с логистическим отображением---')
    print('сходства:')
    print('- наличие бифуркаций')
    print('- переход к сложному поведению при больших r')
    print('различия:')
    print('- разные диапазоны параметров r')
    print('- разные точки бифуркаций')
    print('- вариант 4 имеет кубическую нелинейность (более сложная динамика)')

#---expert level---

# анализ устойчивости неподвижных точек
def stability_analysis():
    print('\n---анализ устойчивости---')

    # анализ для логистического отображения
    print('---логистическое отображение---')
    r_values = [0.5, 1.5, 2.5, 3.0, 3.5]

    for r in r_values:
        points = find_logistic_fixed_points(r)
        print(f'\n r = {r}:')
        for x_star in points:
            deriv = derivative_logistic(x_star, r)
            stability = 'устойчива' if abs(deriv) < 1 else 'неустойчива'
            print(f' x* = {x_star:.4f}, f\'(x*) = {deriv:.4f} - {stability}')

    # для варианта 4
    print('\n---вариант 4---')
    for r in r_values:
        points = analyze_fixed_points(r)
        print(f'\n r = {r}:')
        for x_star in points:
            if x_star > 0:  # не рассматриваем x = 0 отдельно
                deriv = derivative_variant(x_star, r)
                stability = 'устойчива' if abs(deriv) < 1 else 'неустойчива'
                print(f' x* = {x_star:.4f}, g\'(x*) = {deriv:.4f} - {stability}')

# демонстрация чувствительности к начальным условиям (эффект бабочки)
def plot_sensitive_trajectories(x0: float = 0.3, epsilon: float = 1e-5, r: float = 4.0, n_iter: int = 50):
    print(f'\n---чувствительность к начальным условиям (r = {r})---')
    
    y0 = x0 + epsilon
    
    # генерируем траектории
    x_traj = [x0]
    y_traj = [y0]
    
    for i in range(n_iter):
        x_traj.append(logistic_map(x_traj[-1], r))
        y_traj.append(logistic_map(y_traj[-1], r))
    
    # вычисляем расхождение
    divergence = [abs(x - y) for x, y in zip(x_traj, y_traj)]
    
    plt.figure(figsize=(12, 8))
    
    # траектории
    plt.subplot(2, 1, 1)
    plt.plot(range(n_iter+1), x_traj, 'b-', label=f'x₀ = {x0}', linewidth=2)
    plt.plot(range(n_iter+1), y_traj, 'r--', label=f'y₀ = {y0:.2e}', linewidth=2)
    plt.title(f'траектории при r = {r} (эффект бабочки)')
    plt.xlabel('n')
    plt.ylabel('$x_n$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # расхождение
    plt.subplot(2, 1, 2)
    plt.semilogy(range(n_iter+1), divergence, 'g-', linewidth=2)
    plt.title('расхождение траекторий')
    plt.xlabel('n')
    plt.ylabel('|x_n - y_n|')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensitive_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'начальное расхождение: {epsilon:.2e}')
    print(f'конечное расхождение: {divergence[-1]:.6f}')
    print(f'увеличение в {divergence[-1]/epsilon:.2e} раз')

# детальная бифуркационная диаграмма с окнами периодичности
def plot_bifurcation_diagram_detailed():
    print('\n---детальная бифуркационная диаграмма---')
    
    # для логистического отображения
    r_range_log = np.linspace(2.5, 4.0, 800)
    # для варианта 4
    r_range_var = np.linspace(2.0, 6.0, 800)
    
    n_iter = 1000
    burn_in = 800
    x0 = 0.3
    
    plt.figure(figsize=(15, 10))
    
    # логистическое отображение
    plt.subplot(2, 2, 1)
    x_points_log = []
    r_points_log = []
    
    for r in r_range_log:
        x = x0
        # "прогреваем" систему
        for _ in range(burn_in):
            x = logistic_map(x, r)
        # собираем установившиеся значения
        for _ in range(50):
            x = logistic_map(x, r)
            x_points_log.append(x)
            r_points_log.append(r)
    
    plt.plot(r_points_log, x_points_log, 'b,', alpha=0.3, markersize=0.1)
    plt.title('бифуркационная диаграмма\nлогистическое отображение')
    plt.xlabel('r')
    plt.ylabel('x')
    plt.grid(True, alpha=0.3)
    
    # вариант 4
    plt.subplot(2, 2, 2)
    x_points_var = []
    r_points_var = []
    
    for r in r_range_var:
        x = x0
        # "прогреваем" систему
        for _ in range(burn_in):
            x = variant_map(x, r)
        # собираем установившиеся значения
        for _ in range(50):
            x = variant_map(x, r)
            x_points_var.append(x)
            r_points_var.append(r)
    
    plt.plot(r_points_var, x_points_var, 'r,', alpha=0.3, markersize=0.1)
    plt.title('бифуркационная диаграмма\nвариант 4')
    plt.xlabel('r')
    plt.ylabel('x')
    plt.grid(True, alpha=0.3)
    
    # окна периодичности для логистического отображения
    plt.subplot(2, 2, 3)
    r_window = np.linspace(3.82, 3.86, 400)  # окно периода 3
    
    x_points_win = []
    r_points_win = []
    
    for r in r_window:
        x = 0.3
        for _ in range(500):
            x = logistic_map(x, r)
        for _ in range(100):
            x = logistic_map(x, r)
            x_points_win.append(x)
            r_points_win.append(r)
    
    plt.plot(r_points_win, x_points_win, 'b,', alpha=0.5, markersize=0.5)
    plt.title('окно периода 3\n(логистическое отображение)')
    plt.xlabel('r')
    plt.ylabel('x')
    plt.grid(True, alpha=0.3)
    
    # фрактальная структура - увеличение
    plt.subplot(2, 2, 4)
    r_fractal = np.linspace(3.848, 3.856, 300)  # увеличенный фрагмент
    
    x_points_frac = []
    r_points_frac = []
    
    for r in r_fractal:
        x = 0.3
        for _ in range(600):
            x = logistic_map(x, r)
        for _ in range(100):
            x = logistic_map(x, r)
            x_points_frac.append(x)
            r_points_frac.append(r)
    
    plt.plot(r_points_frac, x_points_frac, 'g,', alpha=0.6, markersize=0.5)
    plt.title('фрактальная структура\n(увеличение)')
    plt.xlabel('r')
    plt.ylabel('x')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('expert_bifurcation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # вывод информации об окнах периодичности
    print('\n---окна периодичности в логистическом отображении---')
    print('r ≈ 3.83 - цикл периода 3')
    print('r ≈ 3.74 - цикл периода 5')  
    print('r ≈ 3.63 - цикл периода 6')
    print('r ≈ 3.96 - цикл периода 4')
    print('r ≈ 3.90 - цикл периода 8')

# теоретические доказательства устойчивости
def theoretical_proofs():
    print('\n---теоретические доказательства---')
    
    print('\n1. устойчивость x* = 0 при r ∈ (0,1) для логистического отображения:')
    print('   f(x) = r*x*(1-x), f(0) = 0')
    print('   f\'(x) = r*(1-2x), f\'(0) = r')
    print('   |f\'(0)| = |r| < 1 при r ∈ (0,1) ⇒ устойчивость')
    
    print('\n2. устойчивость x* = 0 для варианта 4 при r ∈ [0, 0.5):')
    print('   g(x) = r*x*(1-x)*(2-x), g(0) = 0')
    print('   g\'(x) = r*(2 - 6x + 3x²), g\'(0) = 2r')
    print('   |g\'(0)| = |2r| < 1 при r < 0.5 ⇒ устойчивость')
    
    print('\n3. неустойчивость x* = 0 при r ∈ (2,3) для логистического отображения:')
    print('   f\'(0) = r > 1 при r > 1 ⇒ неустойчивость')
    
    print('\n4. теорема о цикле периода 3 (ли и йорк):')
    print('   наличие цикла периода 3 влечет хаотическое поведение')
    print('   это подтверждается окном периода 3 при r ≈ 3.83')

# полный анализ expert level
def expert_level_analysis():
    print('\n' + '='*60)
    print('expert level')
    print('='*60)
    
    # анализ устойчивости
    stability_analysis()
    
    # чувствительность к начальным условиям
    plot_sensitive_trajectories()
    
    # детальные бифуркационные диаграммы
    plot_bifurcation_diagram_detailed()
    
    # теоретические доказательства
    theoretical_proofs()
    
    # дополнительный анализ для варианта 4
    print('\n---дополнительный анализ для варианта 4---')
    print('кубическая нелинейность приводит к:')
    print('- более сложной бифуркационной структуре')
    print('- расширенному диапазону параметра r')
    print('- возможности возникновения дополнительных циклов')

# ---основная функция---

def main():
    print('---лабораторная работа по математическому анализу---')
    print('вохмяков данила никитич (504499)')
    print('вариант 4: g(x_{n+1}) = r * x_n * (1 - x_n) * (2 - x_n)')
    
    # easy level
    print('\n' + '='*50)
    print('easy level')
    print('='*50)
    
    print('\n---анализ неподвижных точек---')
    test_r = [0.5, 1.0, 1.5, 2.0, 2.5]
    for r in test_r:
        points = analyze_fixed_points(r)
        print(f'r = {r}: {[f"{p:.4f}" for p in points]}')
    
    print('\n---построение графиков easy level---')
    plot_comparison()
    
    # normal level
    print('\n' + '='*50)
    print('normal level')
    print('='*50)
    
    convergence_analysis()
    monotonicity_analysis()
    
    print('\n---построение графиков normal level---')
    plot_sequences()
    
    # hard level
    print('\n' + '='*50)
    print('hard level')
    print('='*50)
    
    print('\n---лестницы ламерея---')
    plot_lameyre_staircase(r=2.0, x0=0.2, n_iter=10)  # сходимость к точке
    plot_lameyre_staircase(r=3.2, x0=0.2, n_iter=20)  # цикл периода 2
    plot_lameyre_staircase(r=3.5, x0=0.2, n_iter=40)  # цикл периода 4
    
    # лестницы для варианта 4
    plot_lameyre_staircase(r=2.0, x0=0.2, n_iter=10, use_variant=True)
    plot_lameyre_staircase(r=4.0, x0=0.2, n_iter=20, use_variant=True)
    
    # анализ циклов
    analyze_cycles()
    variant_cycle_analysis()
    
    # expert level
    expert_level_analysis()
    
    print('\n' + '='*60)
    print('лабораторная работа завершена')
    print('='*60)
    print('результаты сохранены в файлах:')
    print('- easy_level_results.png')
    print('- normal_level_sequences.png') 
    print('- lameyre_*.png')
    print('- cycle_analysis.png')
    print('- variant_cycles.png')
    print('- expert_bifurcation.png')
    print('- sensitive_trajectories.png')

if __name__ == '__main__':
    main()