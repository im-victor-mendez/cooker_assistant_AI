# Cooker Assistant AI

Prototipo de Inteligencia ARtificial para ayudar a cocinar.

## Idea Principal

> Le indicas los ingredientes que crees vayan bien juntos, la IA busca matches y devuelve la receta como instrucciones interactivas.

## Como colaborar

1. Instalar [Python](https://www.python.org/downloads/) y [pip](https://pip.pypa.io/en/stable/installation/)
2. Instalar librerías necesarias

   1. [Numpy](https://numpy.org/)

      ```console
      pip install numpy
      ```

   2. [TensorFlow](https://www.tensorflow.org/)

      ```console
      pip install tensorflow
      ```

   3. [NLTK](https://www.nltk.org/)

      ```console
      pip install nltk
      ```

3. Puedes agregar _tags_ en el archivo **_model/intents.json_** siguiendo la siguiente estructura dentro de la key _intents_:

   ```json
   {
   	"tag": "Identificador de las interacciones",
   	// Se recomienda que se escriban con inicial minúscula
   	"patterns": [
   		"Patrones para",
   		"ser identificadas",
   		"y regresar una respuesta",
   		"adecuada"
   	],
   	"responses": ["Respuestas que", "quieres que regrese", "el modelo."]
   }
   ```

4. Ejecutar el archivo **_model/training.py_** para generar los archivos **_model/classes.pkl_**, **_model/words.pkl_** y **_model/model.h5_**

5. Ejecutar **_model/chat.py_**

> Por cada modificación en el archivo **_model/intents.json_** sera necesaria la ejecución del archivo **_model/training.py_**.
