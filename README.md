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

3. Puedes consultar recetas creando la Base de Datos ejecutando el archivo **_Database.db_** y agregándole datos.

4. Para tener mas respuestas y mejorar interacciones puedes agregar _tags_ en el archivo **_model/intents.json_** siguiendo la siguiente estructura dentro de la key _intents_:

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

   Para responder con una receta se tiene que seguir la siguiente estructura:

   ```json
   {
   	"tag": "Identificador de las interacciones",
   	// Se recomienda que sea en minúsculas
   	"patterns": ["nombre de ingrediente", "también puede ser en plural"],
   	// Tienen que ser solo consultas sql SELECT
   	// Se propone la siguiente
   	"responses": [
   		"SELECT R.Nombre AS Receta, R.TiempoPreparacion AS 'Tiempo de Preparacion', R.Pasos, R.Rendimiento, R.TamanoPorcion AS 'Tamaño de la Porción', R.TipoPlatillo AS 'Tipo de Platillo', R.Notas, I.Nombre AS 'Nombre del Ingrediente', I.UnidadMedida, I.Categoria, U.Nombre AS 'Nombre del Utensilio' FROM Recetas R JOIN RecetaIngredientes RI ON R.Id = RI.IdReceta JOIN Ingredientes I ON RI.IdIngrediente = I.Id JOIN RecetaUtensilios RU ON R.Id = RU.IdReceta JOIN Utensilios U ON RU.IdUtensilio = U.Id WHERE I.Nombre = 'NOMBRE DEL INGREDIENTE';"
   		// El [NOMBRE DEL INGREDIENTE] debe ser igual a como lo agregaste a la base de datos
   		// Por ejemplo 'Orange'
   	]
   }
   ```

   > Por cada modificación en el archivo **_model/intents.json_** sera necesaria la ejecución del archivo **_model/training.py_**.

5. Ejecutar el archivo **_model/training.py_** para generar los archivos **_model/classes.pkl_**, **_model/words.pkl_** y **_model/model.h5_**

6. Asegurarte que los datos para poder conectarte a tus bases de datos sean los mismos a los que están en el archivo **_model/chat.py_** en la linea 22:

   ```python
   # Conectar a la base de datos
   conn = mysql.connector.connect(
      host="127.0.0.1",
      user="root",
      password="",
      database="Pinche"
   )
   ```

7. Ahora puedes utilizar el chat ejecutando el archivo **_model/chat.py_**
