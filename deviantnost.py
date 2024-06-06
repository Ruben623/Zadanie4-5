import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Входные переменные
academic_performance = ctrl.Antecedent(np.arange(0, 101, 1), 'academic_performance')
school_behavior = ctrl.Antecedent(np.arange(0, 101, 1), 'school_behavior')
family_situation = ctrl.Antecedent(np.arange(0, 101, 1), 'family_situation')

# Выходная переменная
deviance_level = ctrl.Consequent(np.arange(0, 101, 1), 'deviance_level')

# Успеваемость
academic_performance['low'] = fuzz.trimf(academic_performance.universe, [0, 0, 50])
academic_performance['medium'] = fuzz.trimf(academic_performance.universe, [25, 50, 75])
academic_performance['high'] = fuzz.trimf(academic_performance.universe, [50, 100, 100])

# Поведение в школе
school_behavior['poor'] = fuzz.trimf(school_behavior.universe, [0, 0, 50])
school_behavior['average'] = fuzz.trimf(school_behavior.universe, [25, 50, 75])
school_behavior['good'] = fuzz.trimf(school_behavior.universe, [50, 100, 100])

# Семейная ситуация
family_situation['unstable'] = fuzz.trimf(family_situation.universe, [0, 0, 50])
family_situation['average'] = fuzz.trimf(family_situation.universe, [25, 50, 75])
family_situation['stable'] = fuzz.trimf(family_situation.universe, [50, 100, 100])

# Уровень девиантности
deviance_level['low'] = fuzz.trimf(deviance_level.universe, [0, 0, 50])
deviance_level['medium'] = fuzz.trimf(deviance_level.universe, [25, 50, 75])
deviance_level['high'] = fuzz.trimf(deviance_level.universe, [50, 100, 100])

# Определение правил вывода
rule1 = ctrl.Rule(academic_performance['low'] & school_behavior['poor'] & family_situation['unstable'], deviance_level['high'])
rule2 = ctrl.Rule(academic_performance['low'] & school_behavior['poor'] & family_situation['average'], deviance_level['medium'])
rule3 = ctrl.Rule(academic_performance['medium'] & school_behavior['average'] & family_situation['stable'], deviance_level['low'])
rule4 = ctrl.Rule(academic_performance['high'] & school_behavior['good'] & family_situation['stable'], deviance_level['low'])

# Создание системы управления
deviance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
deviance_simulation = ctrl.ControlSystemSimulation(deviance_ctrl)

# Ввод данных для тестирования
deviance_simulation.input['academic_performance'] = 30
deviance_simulation.input['school_behavior'] = 40
deviance_simulation.input['family_situation'] = 20

# Вычисление
deviance_simulation.compute()

# Результат
print(f"Level of deviance: {deviance_simulation.output['deviance_level']}")

# Визуализация функций принадлежности и результатов
academic_performance.view()
school_behavior.view()
family_situation.view()
deviance_level.view(sim=deviance_simulation)
plt.show()
