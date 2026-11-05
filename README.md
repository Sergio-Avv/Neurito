# Neurito
1. **Neurito decide qué hacer:** una mascota que escoge entre comer, dormir o jugar según lo que siente.
2. **Neurito aprende a saltar:** un personaje que debe decidir en qué momento brincar para esquivar obstáculos.

---

## Parte 1. Neurito decide qué hacer

### Puesta en marcha

Necesitas Python 3.12 o superior y la herramienta `uv`. Desde la carpeta del proyecto:

```bash
uv run python main.py
```

Por defecto los textos de la ventana salen en inglés (`LANGUAGE = "en"` en el código). Si prefieres verlos en español, edita esa constante y pon `"es"`.

**Teclas disponibles**

- `H` / `J`: aumenta / disminuye el hambre.
- `E` / `D`: aumenta / disminuye la energía.
- `A` / `S`: aumenta / disminuye el aburrimiento.
- `R`: asigna valores aleatorios a las tres necesidades.
- `N`: regresa al estado de calma.
- `Esc`: cierra el programa.

### Qué hay detrás de cada decisión

La mascota percibe tres necesidades, cada una expresada como un número entre 0 y 1. Para entrenar la red se fabrican 3000 casos aleatorios, y a cada uno se le asigna como respuesta correcta la acción de la necesidad más urgente:

- **Comer** (`eat`): se compara el hambre.
- **Dormir** (`sleep`): se compara el cansancio, es decir, 1 - energía.
- **Jugar** (`play`): se compara el aburrimiento.

Dicho con las entradas `[hambre, energía, aburrimiento]`, la respuesta correcta es la posición del valor más alto dentro de `[hambre, 1 - energía, aburrimiento]`. Cuando otra necesidad pasa a ser la mayor, la acción cambia. Si dos valores están empatados o muy cerca, un movimiento mínimo basta para que cambie la decisión. La red no conoce esta fórmula: la aproxima a partir de los ejemplos.

**Un detalle de la interfaz:** si el hambre y el aburrimiento están por debajo de 0.4 y la energía supera 0.6, la pantalla muestra el estado `neutral`. Esto es solo una regla visual; el modelo sigue calculando probabilidades para las tres acciones y `neutral` nunca se entrenó como clase.

### Resultados de los experimentos

Para que las pruebas fueran comparables se dejaron fijos los datos sintéticos de entrenamiento, la semilla (42) y el descenso por gradiente con lotes completos que usa el programa. Además se evaluó siempre con el mismo conjunto de 1000 ejemplos, generado aparte con la semilla 123. En el código original ese conjunto sale del generador aleatorio que queda después de inicializar los pesos, así que cambiaría al modificar el tamaño de la capa; separarlo evita ese problema. La pérdida que se reporta es la entropía cruzada final calculada sobre los 3000 ejemplos de entrenamiento. Los números pueden variar si se modifican los datos, las semillas, las etiquetas o la implementación.

#### Dos necesidades quietas, una que se mueve

Con energía en 0.50 (cansancio 0.50) y aburrimiento en 0.10, se hizo variar el hambre en el modelo base de 6 neuronas ocultas:

- Hambre 0.49: comer 49.3%, dormir 50.7%. La red elige **dormir**.
- Hambre 0.50: comer 66.9%, dormir 33.1%. La red elige **comer**.
- Hambre 0.52: comer 89.7%, dormir 10.3%. La red elige **comer**.

El cambio se produce alrededor del empate entre hambre y cansancio (0.50 contra 0.50). Como la red solo aproxima la regla y aprende de una cantidad limitada de datos, su punto de giro no tiene por qué caer exactamente en el empate matemático. Cuando las prioridades están bien separadas, la decisión es mucho más estable.

#### Prioridades parecidas y Softmax

Softmax toma los tres valores que entrega la red y los convierte en números entre 0 y 1 cuya suma es 1. Si dos salidas son cercanas, ambas obtienen una probabilidad considerable; al inclinar un poco las entradas hacia una de ellas, su probabilidad crece y la de las demás cae. En el ejemplo anterior, subir el hambre de 0.49 a 0.50 volteó la decisión, y con 0.52 la probabilidad de comer llegó a 89.7%. Hay que leer esas probabilidades como la confianza relativa del modelo entre sus tres opciones; no miden las necesidades de verdad ni aseguran que la acción sea la acertada.

#### Cuántas neuronas ocultas conviene usar

Solo se varió el número de neuronas de la capa oculta, manteniendo 1200 épocas y tasa de aprendizaje 0.1:

- 3 neuronas: pérdida 0.1618, precisión de prueba 96.3%.
- 6 neuronas (base): pérdida 0.1225, precisión de prueba 99.4%.
- 10 neuronas: pérdida 0.1262, precisión de prueba 99.1%.
- 20 neuronas: pérdida 0.1220, precisión de prueba 99.0%.

Con tres neuronas el modelo se queda corto respecto al base en esta ejecución. Con diez o veinte no se gana precisión de prueba: tener más capacidad no asegura generalizar mejor, sobre todo si la regla que se quiere aprender es sencilla.

#### Efecto de las épocas y de la tasa de aprendizaje

Cada comparación modifica un único parámetro y deja el resto en su valor base (6 neuronas, y 1200 épocas o tasa 0.1 según el caso).

- 200 épocas: pérdida 0.4984, precisión 91.2%. El entrenamiento no alcanza.
- 1200 épocas: pérdida 0.1225, precisión 99.4%. Gran mejora respecto a 200.
- 3000 épocas: pérdida 0.0711, precisión 99.7%. La pérdida sigue cayendo, la precisión casi no sube.
- Tasa 0.01: pérdida 0.7376, precisión 80.1%. Aprende muy despacio con 1200 épocas.
- Tasa 0.1: pérdida 0.1225, precisión 99.4%. Punto de referencia.
- Tasa 0.5: pérdida 0.0484, precisión 99.8%. La mejor de las probadas.
- Tasa 1.0: pérdida 0.0447, precisión 98.4%. La pérdida baja, pero la precisión de prueba empeora.
- Tasa 2.0: pérdida 0.0533, precisión 95.9%. Peor resultado: el paso es demasiado grande.

Con estos valores, 1200 épocas ya logran una precisión alta y pasar a 3000 casi no aporta. Una tasa muy pequeña no alcanza a ajustar el modelo en el mismo número de épocas, y una muy grande puede saltarse ajustes útiles y generalizar peor. Esto no demuestra que exista una tasa ideal para todos los casos; para concluirlo con más seguridad habría que repetir con varias semillas y separar datos de validación y de prueba.

### ¿Cómo agregar una cuarta acción?

Primero hay que decidir qué significa la nueva acción y en qué situaciones sería la respuesta correcta. Luego, reunir ejemplos con las tres entradas actuales (o sumar una variable nueva si la acción depende de algo que el modelo hoy no ve) y etiquetarlos con cuatro clases, con suficientes casos representativos de la nueva. A partir de ahí se modifican la regla que genera las etiquetas, la salida de la red (de 3 a 4 valores), la lista de acciones, los textos y la interfaz. También se necesita un conjunto de prueba con las cuatro clases para comprobar que la nueva se reconoce sin confundir las anteriores. Y si dos situaciones con las mismas tres entradas exigen acciones diferentes, será necesario aportar más información de entrada para poder distinguirlas.

### Preguntas para reflexionar

**1. ¿Qué diferencia hay entre pesos y sesgos?**

Los pesos (`W1`, `W2`) determinan cuánto pesa cada entrada, o cada neurona de la capa previa, en la capa siguiente. Los sesgos (`b1`, `b2`) se suman a esa combinación ponderada y mueven el umbral a partir del cual una neurona se activa. Sin ellos, la red tendría menos libertad para tomar decisiones.

**2. ¿Por qué el cansancio se calcula como `1 - energía`?**

Porque ambas magnitudes van en sentidos contrarios: mucha energía equivale a poco cansancio y poca energía a mucho. Al estar la energía entre 0 y 1, restarla de 1 invierte la escala sin salirse de ese rango. Por ejemplo, una energía de 0.2 da un cansancio de 0.8. Es una simplificación de esta demostración, no un modelo del cuerpo real.

**3. ¿Para qué sirven ReLU y Softmax?**

- **ReLU** (`max(0, x)`) se aplica en la capa oculta: conserva los valores positivos y vuelve cero los negativos. Introduce una no linealidad que permite aprender reglas más flexibles que una simple transformación lineal.
- **Softmax** se aplica a la salida: convierte los tres valores finales en probabilidades relativas que suman 1. La predicción es la clase con mayor probabilidad (`argmax`).

**4. ¿Por qué una precisión alta puede engañar?**

Porque solo cuenta cuántas predicciones coinciden con las etiquetas. En este proyecto, tanto el entrenamiento como la prueba se generan al azar con la misma regla artificial, de modo que un 99.4% muestra que la red copia bien esa regla inventada, no que comprenda lo que necesita una mascota real. Además, un promedio general puede esconder fallos en clases poco frecuentes o en los casos cercanos al empate. Evaluar un caso real exigiría datos representativos, etiquetas bien justificadas, métricas por clase y pruebas con datos distintos de los usados para ajustar el modelo.

---

## Parte 2. Neurito aprende a saltar

En esta segunda parte, Neurito debe descubrir **en qué momento saltar** frente a obstáculos que llegan a distintas velocidades. La red es de tipo 3 → 12 → 1, programada a mano con NumPy, y su proceso de aprendizaje se ve en vivo gracias a Pygame.

- **Tipo de aprendizaje:** supervisado, con datos sintéticos (no es aprendizaje por refuerzo).
- **Entradas:** distancia al obstáculo, velocidad y tiempo estimado para alcanzarlo.
- **Arquitectura:** 12 neuronas ocultas con ReLU y una neurona de salida con sigmoide; Neurito salta cuando esa salida pasa de 0.5.
- **Entrenamiento:** entropía cruzada binaria, retropropagación y descenso de gradiente.

### Archivos que se añadieron

- `saltar.py`: el juego y la visualización del entrenamiento.
- `salto_modelo.py`: la red neuronal hecha a mano y los datos de entrenamiento.
- `salto_juego.py`: la física del salto y la detección de choques.
- `GUIA_SALTO.md`: guía detallada y experimentos.

Guía completa: [GUIA_SALTO.md](neurito/GUIA_SALTO.md)

### Cómo jugarlo

Desde la carpeta que contiene `pyproject.toml`:

```bash
uv sync
uv run python saltar.py
```

No hay que instalar nada adicional, porque se aprovechan NumPy y Pygame que ya usa el proyecto.

### Teclas del juego

- `1`: entrena una sola época.
- `2`: entrena cien épocas.
- `3`: enciende o pausa el entrenamiento continuo.
- `A`: cambia entre modo automático y manual.
- `Espacio`: salta de forma manual.
- `C`: reinicia la pista y los contadores.
- `R`: reinicia los pesos y los contadores.
- `Esc`: cierra el juego.

### Pregunta clave

**¿Cómo es posible que una red clasifique bien los ejemplos y aun así choque?**

La precisión indica cuántas decisiones de la red coinciden con las etiquetas de un conjunto de ejemplos; no dice si Neurito logra sobrevivir dentro del juego. Dos motivos principales:

1. **La respuesta "correcta" la definimos nosotros.** Etiquetamos como "saltar" los casos en que faltan entre 7 y 13 fotogramas para llegar al obstáculo. La red aprende muy bien esa ventana, pero nada garantiza que coincida con lo que realmente exige la física del salto.
2. **La física es independiente de la red.** La gravedad, el impulso, la altura alcanzada y el tamaño del obstáculo son los que determinan si Neurito pasa. Una decisión acertada según la etiqueta puede resultar demasiado temprana o demasiado tarde para librar el obstáculo.

**En resumen:** una precisión alta significa que la red aprendió bien la regla de las etiquetas, pero no asegura que esa regla sea buena para esquivar obstáculos con la física real del juego.
