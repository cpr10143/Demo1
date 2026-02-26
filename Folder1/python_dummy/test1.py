import sys
from unittest import TestCase

class Evaluate(TestCase):
    def test_exercise(self):
        import exercise  # Imports and runs student's solution
        output = sys.stdout.getvalue().replace(" ","")  # Returns output since this function started
        self.assertEqual("MeencantaestudiarPython\n", output, "Debes imprimir Me encanta estudiar Python")


import sys
from unittest import TestCase
import io

class Evaluate(TestCase):
    def test_exercise(self):
        sys.stdin = io.StringIO(' ')
        import exercise
        output = sys.stdout.getvalue()
        output = output.replace(" ","")
        self.assertEqual('¿Quéestásestudiando?\n', output,
                         'Debes preguntarle al usuario "¿Qué estás estudiando?", e imprimir su respuesta')