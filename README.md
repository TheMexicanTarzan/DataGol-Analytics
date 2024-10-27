# DataGol-Analytics

**Descripción:**  
DataGol es un proyecto orientado a la categorización de jugadores de fútbol en base a su estilo de juego. El objetivo principal es identificar una alineación óptima que, dada la alineación del oponente, maximice la esperanza de goles a favor. Este análisis se realiza independientemente del rol típico de los jugadores y se basa en estadísticas detalladas de sus acciones en el campo.

---

## Estructura del Proyecto

1. **Importación de Librerías**:  
   Se utilizan bibliotecas para la carga, análisis, y visualización de datos, tales como `mplsoccer`, `statsbombpy`, `scipy`, `pandas`, `numpy`, y `plotly`. Esto asegura una gestión robusta de datos y gráficos especializados en el contexto del fútbol.

2. **Carga de Datos**:  
   La fuente de datos principal es `statsbombpy`, que se utiliza para obtener información detallada de competiciones, partidos y alineaciones de la Copa América. Esta información incluye eventos por partido y alineaciones detalladas de equipos específicos.

3. **Análisis Exploratorio**:  
   Se realiza una inspección preliminar de los datos de competiciones y eventos de la Copa América para entender mejor los patrones y la estructura de los datos.

4. **Modelado**:  
   Se emplean distintas técnicas de machine learning, incluyendo:
   - **Clustering Jerárquico con Método de Ward**
   - **Regresión lineal Gaussiana**
   - **Support Vector Machine**
   - **Modelado con redes neuronales densas**

5. **Resultados**:
   Los modelos anteriores tienen como resultado lo siguiente:
   - **Categorías de jugadores basado en sus personalidades de juego**
   - **Esperanza de goles en un partido**
   - **Cambio óptimo de una alineación para maximizar goles esperados**

6.**Conclusion**:
   Recapitulación del proyecto y detalle de pasos futuros a tomar para mejorarlo.
   

## Requisitos

Para ejecutar el proyecto, se deben instalar las siguientes dependencias:
```bash
pip install mplsoccer statsbombpy scipy pandas numpy plotly tensorflow seaborn scikit-learn
