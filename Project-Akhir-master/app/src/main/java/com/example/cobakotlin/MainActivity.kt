package com.example.cobakotlin

import android.content.ActivityNotFoundException
import android.content.Intent
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.android.volley.Request
import com.android.volley.RequestQueue
import com.android.volley.Response
import com.android.volley.toolbox.JsonObjectRequest
import com.android.volley.toolbox.Volley
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.*

class MainActivity : AppCompatActivity() {

    private val RESULT_SPEECH = 1
    private lateinit var tvVoice: TextView
    private lateinit var tvChat: TextView
    private lateinit var btnVoice: Button
    private val ID_BahasaIndonesia = "id"
    private var textToSpeech: TextToSpeech? = null
    private lateinit var requestQueue: RequestQueue

    private val API_URL = "https://api.openai.com/v1/chat/completions"
    private val API_KEY = "Masukkan Kode API"

    // VARIABEL UNTUK TEXT MINING
    private lateinit var vocabulary: Map<String, Int>
    private lateinit var featureMatrix: Array<DoubleArray>
    private lateinit var labels: IntArray

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvChat = findViewById(R.id.tvChat)
        tvVoice = findViewById(R.id.tvVoice)
        btnVoice = findViewById(R.id.btnVoice)
        requestQueue = Volley.newRequestQueue(this)

        // LOAD MODEL TEXT MINING
        loadTextMiningModel()

        btnVoice.setOnClickListener {
            val micGoogle = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, ID_BahasaIndonesia)
            }

            try {
                textToSpeech?.stop()
                startActivityForResult(micGoogle, RESULT_SPEECH)
                tvVoice.text = ""
            } catch (e: ActivityNotFoundException) {
                Toast.makeText(applicationContext, "Maaf, Device Kamu Tidak Support Speech To Text", Toast.LENGTH_SHORT).show()
                e.printStackTrace()
            }
        }
    }

    // FUNGSI LOAD MODEL TEXT MINING
    private fun loadTextMiningModel() {
        try {
            vocabulary = loadVocabulary()
            featureMatrix = loadFeatureMatrix()
            labels = loadLabels()
            Toast.makeText(this, "Model text mining loaded successfully!", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }

    // LOAD VOCABULARY
    private fun loadVocabulary(): Map<String, Int> {
        val inputStream = assets.open("vocab.json")
        val size = inputStream.available()
        val buffer = ByteArray(size)
        inputStream.read(buffer)
        inputStream.close()
        val json = String(buffer, Charsets.UTF_8)

        val jsonObject = JSONObject(json)
        val vocabMap = mutableMapOf<String, Int>()

        val keys = jsonObject.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            vocabMap[key] = jsonObject.getInt(key)
        }
        return vocabMap
    }

    // LOAD FEATURE MATRIX
    private fun loadFeatureMatrix(): Array<DoubleArray> {
        val inputStream = assets.open("X_train.csv")
        val reader = BufferedReader(InputStreamReader(inputStream))
        val lines = reader.readLines()
        reader.close()
        inputStream.close()

        return lines.map { line ->
            line.split(",").map { it.toDouble() }.toDoubleArray()
        }.toTypedArray()
    }

    // LOAD LABELS
    private fun loadLabels(): IntArray {
        val inputStream = assets.open("y_train.csv")
        val reader = BufferedReader(InputStreamReader(inputStream))
        val lines = reader.readLines()
        reader.close()
        inputStream.close()

        return lines.map { it.toDouble().toInt() }.toIntArray()
    }

    // PREPROCESSING TEKS
    private fun preprocessText(text: String): List<String> {
        return text.toLowerCase()
            .replace(Regex("[^a-zA-Z0-9\\s]"), "")
            .split("\\s+".toRegex())
            .filter { it.isNotEmpty() }
    }

    // EKSTRAKSI FITUR
    private fun extractFeatures(text: String): DoubleArray {
        val tokens = preprocessText(text)
        val features = DoubleArray(vocabulary.size) { 0.0 }

        for (token in tokens) {
            vocabulary[token]?.let { index ->
                features[index] += 1.0
            }
        }

        // Normalisasi
        val magnitude = Math.sqrt(features.sumOf { it * it })
        if (magnitude > 0) {
            for (i in features.indices) {
                features[i] /= magnitude
            }
        }

        return features
    }

    // KLASIFIKASI KNN
    private fun classifyText(text: String): String {
        try {
            val features = extractFeatures(text)
            val result = knnClassify(features, k = 3)
            return result
        } catch (e: Exception) {
            return "Error: ${e.message}"
        }
    }

    // IMPLEMENTASI KNN
    private fun knnClassify(features: DoubleArray, k: Int = 3): String {
        val distances = mutableListOf<Pair<Double, Int>>()

        for (i in featureMatrix.indices) {
            val distance = calculateEuclideanDistance(features, featureMatrix[i])
            val label = labels[i]
            distances.add(Pair(distance, label))
        }

        distances.sortBy { it.first }
        val kNearest = distances.take(k)

        val labelVotes = mutableMapOf<Int, Int>()
        for ((_, label) in kNearest) {
            labelVotes[label] = labelVotes.getOrDefault(label, 0) + 1
        }

        val predictedLabel = labelVotes.maxByOrNull { it.value }?.key ?: 0

        // SESUAIKAN DENGAN LABEL KAMU
        return when (predictedLabel) {
            0 -> "Kelas 1"
            1 -> "Kelas 2"
            2 -> "Kelas 3"
            else -> "Unknown"
        }
    }

    // HITUNG JARAK EUCLIDEAN
    private fun calculateEuclideanDistance(a: DoubleArray, b: DoubleArray): Double {
        var sum = 0.0
        for (i in a.indices) {
            val diff = a[i] - b[i]
            sum += diff * diff
        }
        return Math.sqrt(sum)
    }

    // MODIFIKASI onActivityResult
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == RESULT_SPEECH && resultCode == RESULT_OK && data != null) {
            val result = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            val spokenText = result?.get(0) ?: ""

            tvVoice.text = spokenText

            // PROSES TEXT MINING
            val classificationResult = classifyText(spokenText)
            tvChat.text = "Hasil Klasifikasi: $classificationResult\n\nTeks: $spokenText"

            // UCAPKAN HASIL
            speak("Hasil klasifikasi: $classificationResult")
        }
    }

    // FUNGSI SEND TO CHATGPT (tetap sama)
    private fun sendToChatGPT(userInput: String) {
        try {
            val requestBody = JSONObject().apply {
                put("model", "gpt-3.5-turbo")
                put("temperature", 0.7)
                val messages = JSONArray().apply {
                    val userMessage = JSONObject().apply {
                        put("role", "user")
                        put("content", userInput)
                    }
                    put(userMessage)
                }
                put("messages", messages)
            }

            val jsonObjectRequest = object : JsonObjectRequest(
                Request.Method.POST,
                API_URL,
                requestBody,
                Response.Listener { response ->
                    try {
                        Log.d("API Response", response.toString())

                        val choices = response.getJSONArray("choices")
                        val chatGPTResponse = choices.getJSONObject(0).getJSONObject("message").getString("content")

                        Toast.makeText(this@MainActivity, chatGPTResponse, Toast.LENGTH_SHORT).show()

                        tvChat.text = chatGPTResponse
                        speak(chatGPTResponse)

                    } catch (e: JSONException) {
                        e.printStackTrace()
                        Toast.makeText(this@MainActivity, "Error parsing response", Toast.LENGTH_SHORT).show()
                    }
                },
                Response.ErrorListener { error ->
                    error.printStackTrace()
                    val statusCode = error.networkResponse?.statusCode ?: -1
                    val errorBody = error.networkResponse?.data?.let { String(it) } ?: "No body"
                    Log.e("API_ERROR", "Status: $statusCode\nBody: $errorBody")
                    Toast.makeText(this@MainActivity, "API Error ($statusCode): $errorBody", Toast.LENGTH_LONG).show()
                }
            ) {
                override fun getHeaders(): MutableMap<String, String> {
                    val headers = HashMap<String, String>()
                    headers["Authorization"] = "Bearer $API_KEY"
                    headers["Content-Type"] = "application/json"
                    return headers
                }
            }

            requestQueue.add(jsonObjectRequest)

        } catch (e: JSONException) {
            e.printStackTrace()
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // FUNGSI SPEAK (tetap sama)
    private fun speak(text: String) {
        if (textToSpeech == null) {
            textToSpeech = TextToSpeech(this) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    textToSpeech?.language = Locale("id", "ID")
                }
            }
        }
        textToSpeech?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    override fun onDestroy() {
        textToSpeech?.apply {
            stop()
            shutdown()
        }
        super.onDestroy()
    }
}