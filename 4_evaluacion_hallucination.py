import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Para evitar warnings de tokenizers

import json
import pandas as pd
from datetime import datetime
import groq
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
import re

class RAGEvaluator:
    def __init__(self, chroma_db_path="./chroma_novelas_db", collection_name="novelas_corpus"):
        """Inicializa el evaluador RAG"""
        self.model = SentenceTransformer(
            "jinaai/jina-embeddings-v2-small-en",
            trust_remote_code=True
        )
        
        # Conexión a Chroma
        self.client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        self.collection = self.client.get_collection(collection_name)
        
        # Cliente Groq
        self.groq_client = groq.Client(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Mapeo de novelas para evaluación
        self.novelas_corpus = {
            "La_de_Bringas_314648": "La de Bringas",
            "El_sombrero_de_tres_picos_pg29506": "El sombrero de tres picos",
            "Tristana_pg66979": "Tristana",
            "La_gaviota_pg23600": "La gaviota",
            "Pepita_Jimenez_pg17223": "Pepita Jiménez",
            "Su_unico_hijo_pg17341": "Su único hijo",
            "La_desheredada_pg25956": "La desheredada",
            "Peñas_arriba_pg24127": "Peñas arriba",
            "Platero_y_yo_pg9980": "Platero y yo",
            "Los_pazos_de_Ulloa_18005-8_UTF8": "Los pazos de Ulloa"
        }
    
    def crear_dataset_evaluacion(self):
        """Crea un dataset de evaluación con diferentes tipos de preguntas"""
        
        # TIPO 1: Preguntas que SÍ tienen respuesta en el corpus
        preguntas_validas = [
            {
                "pregunta": "¿Quién es el protagonista de Tristana?",
                "tipo": "especifica",
                "tiene_respuesta": True,
                "novela_objetivo": "Tristana",
                "respuesta_esperada": "Tristana es la protagonista"
            },
            {
                "pregunta": "¿Cómo se llama el burrito en Platero y yo?",
                "tipo": "especifica", 
                "tiene_respuesta": True,
                "novela_objetivo": "Platero y yo",
                "respuesta_esperada": "Platero"
            },
            {
                "pregunta": "¿Qué temas románticos aparecen en las novelas?",
                "tipo": "general",
                "tiene_respuesta": True,
                "novela_objetivo": "múltiples",
                "respuesta_esperada": "Temas de amor, relaciones, etc."
            },
            {
                "pregunta": "¿Cuál es el final de La gaviota?",
                "tipo": "especifica",
                "tiene_respuesta": True,
                "novela_objetivo": "La gaviota",
                "respuesta_esperada": "Información sobre el desenlace"
            },
            {
                "pregunta": "¿Qué diferencias hay entre los personajes de Pepita Jiménez y Tristana?",
                "tipo": "comparativa",
                "tiene_respuesta": True,
                "novela_objetivo": "múltiples",
                "respuesta_esperada": "Comparación de personajes"
            }
        ]
        
        # TIPO 2: Preguntas que NO tienen respuesta (fuera del corpus)
        preguntas_sin_respuesta = [
            {
                "pregunta": "¿Cuál es la opinión de Cervantes sobre Don Quijote?",
                "tipo": "fuera_corpus",
                "tiene_respuesta": False,
                "novela_objetivo": "ninguna",
                "respuesta_esperada": "No sé / No tengo información"
            },
            {
                "pregunta": "¿Qué dice García Márquez sobre el realismo mágico?",
                "tipo": "fuera_corpus",
                "tiene_respuesta": False,
                "novela_objetivo": "ninguna",
                "respuesta_esperada": "No sé / No tengo información"
            },
            {
                "pregunta": "¿Cuándo nació Benito Pérez Galdós?",
                "tipo": "fuera_corpus",
                "tiene_respuesta": False,
                "novela_objetivo": "ninguna",
                "respuesta_esperada": "No sé / No tengo información"
            },
            {
                "pregunta": "¿Qué premio Nobel ganó Juan Ramón Jiménez?",
                "tipo": "fuera_corpus",
                "tiene_respuesta": False,
                "novela_objetivo": "ninguna",
                "respuesta_esperada": "No sé / No tengo información"
            }
        ]
        
        # TIPO 3: Preguntas con información incorrecta (trampa)
        preguntas_trampa = [
            {
                "pregunta": "¿Por qué Tristana es la hermana de Don Quijote?",
                "tipo": "trampa",
                "tiene_respuesta": False,
                "novela_objetivo": "ninguna",
                "respuesta_esperada": "Corrección de la premisa falsa"
            },
            {
                "pregunta": "¿Cómo reacciona Platero cuando conoce a Sancho Panza?",
                "tipo": "trampa",
                "tiene_respuesta": False,
                "novela_objetivo": "ninguna",
                "respuesta_esperada": "Corrección de la premisa falsa"
            },
            {
                "pregunta": "¿En qué año se publicó La de Bringas en el siglo XX?",
                "tipo": "trampa",
                "tiene_respuesta": False,
                "novela_objetivo": "ninguna",
                "respuesta_esperada": "Corrección temporal"
            }
        ]
        
        # TIPO 4: Preguntas ambiguas o vagas
        preguntas_ambiguas = [
            {
                "pregunta": "¿Qué pasa?",
                "tipo": "ambigua",
                "tiene_respuesta": False,
                "novela_objetivo": "ninguna",
                "respuesta_esperada": "Solicitar clarificación"
            },
            {
                "pregunta": "¿Es bueno?",
                "tipo": "ambigua",
                "tiene_respuesta": False,
                "novela_objetivo": "ninguna",
                "respuesta_esperada": "Solicitar clarificación"
            },
            {
                "pregunta": "¿Cuál es mejor?",
                "tipo": "ambigua",
                "tiene_respuesta": False,
                "novela_objetivo": "ninguna",
                "respuesta_esperada": "Solicitar clarificación"
            }
        ]
        
        # Combinar todas las preguntas
        dataset_completo = (preguntas_validas + preguntas_sin_respuesta + 
                          preguntas_trampa + preguntas_ambiguas)
        
        return dataset_completo
    
    def buscar_contexto(self, query: str, top_k: int = 5) -> List[Dict]:
        """Busca contexto relevante en Chroma"""
        query_embedding = self.model.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        contexts = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            contexts.append({
                'texto': doc,
                'novela': metadata['novela'],
                'chunk_index': metadata['chunk_index'],
                'distancia': distance
            })
        
        return contexts
    
    def generar_respuesta(self, query: str, contexts: List[Dict]) -> str:
        """Genera respuesta usando el modelo RAG"""
        
        if not contexts:
            contexto_texto = "No se encontró información relevante en el corpus."
        else:
            contexto_texto = "\n\n".join([f"De {ctx['novela']}: {ctx['texto']}" 
                                        for ctx in contexts])
        
        prompt = f"""
Con base en la siguiente información del corpus de literatura española del siglo XIX:

{contexto_texto}

Responde a la pregunta: {query}

INSTRUCCIONES IMPORTANTES:
- Responde ÚNICAMENTE basándote en la información proporcionada
- Si no tienes información suficiente, di "No tengo suficiente información en el corpus para responder esta pregunta"
- NO inventes detalles que no estén en el texto
- Si la pregunta contiene información incorrecta, corrígela
- Cita de qué novela proviene la información
"""
        
        response = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un experto en literatura española del siglo XIX. Responde únicamente basándote en la información proporcionada. No inventes información."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.1,  # Temperatura muy baja para reducir creatividad
            max_tokens=512
        )
        
        return response.choices[0].message.content
    
    def evaluar_hallucination(self, pregunta: Dict, respuesta: str, contexts: List[Dict]) -> Dict:
        """Evalúa si la respuesta contiene alucinaciones"""
        
        resultado = {
            "pregunta_id": pregunta.get("pregunta"),
            "tipo_pregunta": pregunta.get("tipo"),
            "tiene_respuesta_corpus": pregunta.get("tiene_respuesta"),
            "respuesta_generada": respuesta,
            "contextos_recuperados": len(contexts),
            "puntuaciones": {}
        }
        
        # MÉTRICA 1: Detección de "No sé" para preguntas sin respuesta
        frases_no_se = [
            "no tengo", "no sé", "no encuentro", "no hay información",
            "no puedo responder", "no está disponible", "no dispongo",
            "información insuficiente", "no se menciona"
        ]
        
        contiene_no_se = any(frase in respuesta.lower() for frase in frases_no_se)
        
        if not pregunta.get("tiene_respuesta"):
            # Para preguntas sin respuesta, DEBE decir "no sé"
            resultado["puntuaciones"]["manejo_sin_respuesta"] = 1.0 if contiene_no_se else 0.0
        else:
            # Para preguntas con respuesta, NO debe decir "no sé"
            resultado["puntuaciones"]["respuesta_cuando_debe"] = 0.0 if contiene_no_se else 1.0
        
        # MÉTRICA 2: Similaridad semántica con contexto recuperado
        if contexts:
            textos_contexto = [ctx['texto'] for ctx in contexts]
            contexto_completo = " ".join(textos_contexto)
            
            # Calcular embeddings
            embedding_respuesta = self.model.encode(respuesta)
            embedding_contexto = self.model.encode(contexto_completo)
            
            # Similaridad coseno
            similaridad = cosine_similarity(
                embedding_respuesta.reshape(1, -1),
                embedding_contexto.reshape(1, -1)
            )[0][0]
            
            resultado["puntuaciones"]["similaridad_contexto"] = float(similaridad)
        else:
            resultado["puntuaciones"]["similaridad_contexto"] = 0.0
        
        # MÉTRICA 3: Detección de nombres/datos inventados (CORREGIDA)
        # Buscar nombres propios que no estén en el contexto
        import string
        
        # Mejorar detección: solo nombres que NO estén al inicio de oración
        oraciones = respuesta.split('.')
        nombres_respuesta = set()
        
        for oracion in oraciones:
            oracion = oracion.strip()
            if oracion:
                # Buscar nombres que NO sean la primera palabra de la oración
                palabras = oracion.split()
                for i, palabra in enumerate(palabras[1:], 1):  # Saltar primera palabra
                    # CORRECCIÓN DEL REGEX: cerrar correctamente el patrón
                    if re.match(r'^[A-Z][a-z]+$', palabra) and len(palabra) > 2:
                        nombres_respuesta.add(palabra)
        
        if contexts:
            contexto_completo = " ".join([ctx['texto'] for ctx in contexts])
            nombres_contexto = set()
            
            # Aplicar la misma lógica al contexto
            oraciones_ctx = contexto_completo.split('.')
            for oracion in oraciones_ctx:
                oracion = oracion.strip()
                if oracion:
                    palabras = oracion.split()
                    for i, palabra in enumerate(palabras[1:], 1):
                        # CORRECCIÓN DEL REGEX: cerrar correctamente el patrón
                        if re.match(r'^[A-Z][a-z]+$', palabra) and len(palabra) > 2:
                            nombres_contexto.add(palabra)
            
            nombres_inventados = nombres_respuesta - nombres_contexto
            
            # Filtrar nombres comunes/genéricos EXPANDIDO
            nombres_genericos = {
                'De', 'El', 'La', 'Los', 'Las', 'Don', 'Doña', 'Siglo', 'España', 'Madrid',
                'No', 'Si', 'Por', 'Sin', 'Con', 'Para', 'Desde', 'Hasta', 'Entre',
                'Representar', 'Recitar', 'Escribir', 'Leer', 'Ver', 'Hacer', 'Ser',
                'Arriba', 'Abajo', 'Norte', 'Sur', 'Este', 'Oeste'
            }
            nombres_inventados = nombres_inventados - nombres_genericos
            
            resultado["puntuaciones"]["nombres_inventados"] = len(nombres_inventados)
            resultado["nombres_sospechosos"] = list(nombres_inventados)
        else:
            resultado["puntuaciones"]["nombres_inventados"] = 0
            resultado["nombres_sospechosos"] = []
        
        # MÉTRICA 4: Detección de fechas/datos específicos inventados
        fechas_respuesta = re.findall(r'\b\d{4}\b', respuesta)  # Años
        numeros_especificos = re.findall(r'\b\d+\b', respuesta)  # Números
        
        if contexts:
            contexto_completo = " ".join([ctx['texto'] for ctx in contexts])
            fechas_contexto = set(re.findall(r'\b\d{4}\b', contexto_completo))
            fechas_inventadas = set(fechas_respuesta) - fechas_contexto
            resultado["puntuaciones"]["fechas_inventadas"] = len(fechas_inventadas)
            resultado["fechas_sospechosas"] = list(fechas_inventadas)
        else:
            resultado["puntuaciones"]["fechas_inventadas"] = len(fechas_respuesta)
            resultado["fechas_sospechosas"] = fechas_respuesta
        
        # MÉTRICA 5: Puntuación general de hallucination
        factores_hallucination = []
        
        # Factor 1: Manejo adecuado de preguntas sin respuesta
        if not pregunta.get("tiene_respuesta"):
            factores_hallucination.append(resultado["puntuaciones"]["manejo_sin_respuesta"])
        else:
            factores_hallucination.append(resultado["puntuaciones"]["respuesta_cuando_debe"])
        
        # Factor 2: Similaridad con contexto (invertida para hallucination)
        if contexts:
            factores_hallucination.append(min(1.0, resultado["puntuaciones"]["similaridad_contexto"]))
        
        # Factor 3: Penalización por nombres inventados
        penalizacion_nombres = min(1.0, resultado["puntuaciones"]["nombres_inventados"] * 0.2)
        factores_hallucination.append(max(0.0, 1.0 - penalizacion_nombres))
        
        # Factor 4: Penalización por fechas inventadas
        penalizacion_fechas = min(1.0, resultado["puntuaciones"]["fechas_inventadas"] * 0.3)
        factores_hallucination.append(max(0.0, 1.0 - penalizacion_fechas))
        
        # Puntuación final (promedio de factores)
        resultado["puntuaciones"]["score_general"] = np.mean(factores_hallucination)
        
        return resultado
    
    #def ejecutar_evaluacion(self, guardar_resultados: bool = True) -> Dict:
    def ejecutar_evaluacion(self, directorio_resultados, guardar_resultados: bool = True) -> Dict:

        """Ejecuta la evaluación completa"""
        
        print("  Iniciando evaluación de hallucinations...")
        print("=" * 60)
        
        # Crear dataset
        dataset = self.crear_dataset_evaluacion()
        
        resultados = []
        
        for i, pregunta in enumerate(dataset, 1):
            print(f"\n[{i}/{len(dataset)}] Evaluando: {pregunta['pregunta'][:50]}...")
            
            # Buscar contexto
            contexts = self.buscar_contexto(pregunta['pregunta'])
            
            # Generar respuesta
            respuesta = self.generar_respuesta(pregunta['pregunta'], contexts)
            
            # Evaluar hallucination
            evaluacion = self.evaluar_hallucination(pregunta, respuesta, contexts)
            
            resultados.append(evaluacion)
            
            # Mostrar resultado inmediato
            score = evaluacion["puntuaciones"]["score_general"]
            print(f"  Score: {score:.3f} | Tipo: {pregunta['tipo']}")
            if evaluacion.get("nombres_sospechosos"):
                print(f"  ⚠️  Nombres sospechosos: {evaluacion['nombres_sospechosos']}")
        
        # Calcular estadísticas agregadas
        estadisticas = self.calcular_estadisticas(resultados)
        
        # Guardar resultados
        if guardar_resultados:
            #self.guardar_resultados(resultados, estadisticas)
            self.guardar_resultados(resultados, estadisticas, directorio_resultados)

        
        # Mostrar resumen
        self.mostrar_resumen(estadisticas)
        
        return {
            "resultados_individuales": resultados,
            "estadisticas": estadisticas
        }
    
    def calcular_estadisticas(self, resultados: List[Dict]) -> Dict:
        """Calcula estadísticas agregadas de la evaluación"""
        
        stats = {
            "total_preguntas": len(resultados),
            "por_tipo": {},
            "metricas_generales": {}
        }
        
        # Agrupar por tipo
        by_tipo = {}
        for resultado in resultados:
            tipo = resultado["tipo_pregunta"]
            if tipo not in by_tipo:
                by_tipo[tipo] = []
            by_tipo[tipo].append(resultado)
        
        # Calcular estadísticas por tipo
        for tipo, items in by_tipo.items():
            scores = [item["puntuaciones"]["score_general"] for item in items]
            nombres_inventados = [item["puntuaciones"]["nombres_inventados"] for item in items]
            
            stats["por_tipo"][tipo] = {
                "count": len(items),
                "score_promedio": np.mean(scores),
                "score_std": np.std(scores),
                "nombres_inventados_promedio": np.mean(nombres_inventados),
                "casos_problematicos": len([s for s in scores if s < 0.7])
            }
        
        # Métricas generales
        all_scores = [r["puntuaciones"]["score_general"] for r in resultados]
        all_nombres = [r["puntuaciones"]["nombres_inventados"] for r in resultados]
        
        stats["metricas_generales"] = {
            "score_promedio_global": np.mean(all_scores),
            "score_std_global": np.std(all_scores),
            "nombres_inventados_total": sum(all_nombres),
            "casos_problematicos_total": len([s for s in all_scores if s < 0.7]),
            "porcentaje_casos_problematicos": (len([s for s in all_scores if s < 0.7]) / len(all_scores)) * 100
        }
        
        return stats
    
    #def guardar_resultados(self, resultados: List[Dict], estadisticas: Dict):
    def guardar_resultados(self, resultados: List[Dict], estadisticas: Dict, directorio_resultados):
        """Guarda los resultados en archivos"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar resultados detallados
        #with open(f"evaluacion_hallucination_{timestamp}.json", "w", encoding="utf-8") as f:
        with open(f"{directorio_resultados}/evaluacion_hallucination_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump({
                "resultados": resultados,
                "estadisticas": estadisticas,
                "timestamp": timestamp
            }, f, indent=2, ensure_ascii=False)
        
        # Crear DataFrame para análisis
        df_data = []
        for r in resultados:
            df_data.append({
                "pregunta": r["pregunta_id"],
                "tipo": r["tipo_pregunta"],
                "score_general": r["puntuaciones"]["score_general"],
                "similaridad_contexto": r["puntuaciones"]["similaridad_contexto"],
                "nombres_inventados": r["puntuaciones"]["nombres_inventados"],
                "fechas_inventadas": r["puntuaciones"]["fechas_inventadas"],
                "contextos_recuperados": r["contextos_recuperados"],
                "respuesta": r["respuesta_generada"][:100] + "..." if len(r["respuesta_generada"]) > 100 else r["respuesta_generada"]
            })
        
        df = pd.DataFrame(df_data)
        #df.to_csv(f"evaluacion_hallucination_{timestamp}.csv", index=False, encoding="utf-8")
        df.to_csv(f"{directorio_resultados}/evaluacion_hallucination_{timestamp}.csv", index=False, encoding="utf-8")
        
        print(f"\n📊 Resultados guardados:")
        print(f"  - JSON: evaluacion_hallucination_{timestamp}.json")
        print(f"  - CSV: evaluacion_hallucination_{timestamp}.csv")
    
    def mostrar_resumen(self, estadisticas: Dict):
        """Muestra un resumen de los resultados"""
        
        print("\n" + "=" * 60)
        print("  RESUMEN DE LA EVALUACIÓN")
        print("=" * 60)
        
        general = estadisticas["metricas_generales"]
        
        print(f"🎯 Score General: {general['score_promedio_global']:.3f} (±{general['score_std_global']:.3f})")
        print(f"⚠️  Casos Problemáticos: {general['casos_problematicos_total']}/{estadisticas['total_preguntas']} ({general['porcentaje_casos_problematicos']:.1f}%)")
        print(f"🔍 Nombres Inventados: {general['nombres_inventados_total']}")
        
        print(f"\n📈 Por Tipo de Pregunta:")
        for tipo, stats in estadisticas["por_tipo"].items():
            print(f"  {tipo:15} | Score: {stats['score_promedio']:.3f} | Problemas: {stats['casos_problematicos']}/{stats['count']}")
        
        # Interpretación
        print(f"\n💡 Interpretación:")
        if general['score_promedio_global'] >= 0.8:
            print("  ✅ Excelente control de hallucinations")
        elif general['score_promedio_global'] >= 0.6:
            print("  ⚠️  Control moderado, necesita mejoras")
        else:
            print("  ❌ Control deficiente, requiere atención urgente")

def main():
    """Ejecuta la evaluación"""
    
    directorio_resultados = "./resultados_evaluacion"
    # Crear directorio si no existe
    if not os.path.exists(directorio_resultados):
        os.makedirs(directorio_resultados)

    print(" Iniciando evaluación de hallucinations para Chatbot RAG de novelas")
    
    try:
        evaluador = RAGEvaluator()
        resultados = evaluador.ejecutar_evaluacion(directorio_resultados)
        
        print("\n Evaluación completada exitosamente!")
        
    except Exception as e:
        print(f"x Error durante la evaluación: {e}")
        print("Asegúrate de que:")
        print("- Chroma DB esté disponible")
        print("- GROQ_API_KEY esté configurada")
        print("- Todos los paquetes estén instalados")

if __name__ == "__main__":
    main()