package com.example.reconocimiento_manos

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.floatingactionbutton.FloatingActionButton
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var tvTotalRacimos: TextView
    private lateinit var tvTotalManos: TextView
    private lateinit var tvColorCinta: TextView
    private lateinit var btnResetTracker: FloatingActionButton

    private lateinit var cameraExecutor: ExecutorService
    private var tfLiteInterpreter: Interpreter? = null

    // Instanciamos el tracker inteligente encargado de recordar los IDs únicos al girar el racimo
    private val bananaTracker = BananaTracker()

    // Variable global que recuerda el último estado de color para evitar el parpadeo en la interfaz
    private var ultimoColorDetectado = "Sin Cinta"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        tvTotalRacimos = findViewById(R.id.tvTotalRacimos)
        tvTotalManos = findViewById(R.id.tvTotalManos)
        tvColorCinta = findViewById(R.id.tvColorCinta)
        btnResetTracker = findViewById(R.id.btnResetTracker)

        // Inicializamos los valores por defecto exactos que necesitas ver al arrancar
        tvTotalRacimos.text = "Total de Racimos Enfocados: 0"
        tvTotalManos.text = "Número de Manos Contadas: 0"
        tvColorCinta.text = "Color de la Cinta: $ultimoColorDetectado"

        // CONFIGURACIÓN DEL CLIC DEL BOTÓN DE REINICIO (Para pasar a un racimo nuevo)
        btnResetTracker.setOnClickListener {
            bananaTracker.resetTracker()
            ultimoColorDetectado = "Sin Cinta"

            tvTotalRacimos.text = "Total de Racimos Enfocados: 0"
            tvTotalManos.text = "Número de Manos Contadas: 0"
            tvColorCinta.text = "Color de la Cinta: $ultimoColorDetectado"

            Toast.makeText(this, "Contador reiniciado. Listo para el nuevo racimo.", Toast.LENGTH_SHORT).show()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        initObjectDetector()

        // Esperamos que la vista se asiente en pantalla para validar permisos de cámara de manera segura
        viewFinder.post {
            checkCameraPermission()
        }
    }

    private fun initObjectDetector() {
        try {
            val assetFileDescriptor = assets.openFd("best_float16.tflite")
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            tfLiteInterpreter = Interpreter(modelBuffer, options)
            Log.d("TensorFlow", "¡Modelo YOLOv8 cargado exitosamente mediante Interpreter!")

        } catch (e: Exception) {
            Log.e("TensorFlow", "Error al cargar el modelo .tflite", e)
            runOnUiThread {
                Toast.makeText(this, "Error al cargar el modelo de IA. Revisa el archivo en assets.", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun checkCameraPermission() {
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
                mostrarDialogoExplicativo()
            } else {
                ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val rotation = viewFinder.display?.rotation ?: android.view.Surface.ROTATION_0

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        procesarImagenConIA(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("CameraX", "Fallo el inicio de la cámara", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.camera.core.ExperimentalGetImage
    private fun procesarImagenConIA(imageProxy: ImageProxy) {
        val interpreter = tfLiteInterpreter
        if (interpreter != null) {
            val bitmap = imageProxy.toBitmap()

            if (bitmap != null) {
                val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()) }
                val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                val resizedBitmap = Bitmap.createScaledBitmap(rotatedBitmap, 640, 640, true)

                val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

                // Formato de salida estándar de YOLOv8: 1 lote, 300 detecciones máximas, 6 atributos por caja
                val outputMap = Array(1) { Array(300) { FloatArray(6) } }

                try {
                    interpreter.run(inputBuffer, outputMap)

                    // Lista temporal donde guardaremos las coordenadas de las manos vistas en ESTE cuadro físico
                    val cajasManosCuadroActual = mutableListOf<RectF>()
                    var cintaEncontradaEnEsteFrame = false

                    // Analizamos las 300 cajas de predicción del array del modelo
                    for (i in 0 until 300) {
                        val score = outputMap[0][i][4]
                        val classId = outputMap[0][i][5].toInt()

                        // Umbral de confianza del 50% para dar por válida una detección
                        if (score > 0.5f) {
                            when (classId) {
                                0 -> {
                                    // Clase 0: Corresponde a las MANOS detectadas.
                                    // Extraemos las coordenadas de la caja (X e Y relativas entre 0.0 y 640.0)
                                    val left = outputMap[0][i][0]
                                    val top = outputMap[0][i][1]
                                    val right = outputMap[0][i][2]
                                    val bottom = outputMap[0][i][3]

                                    cajasManosCuadroActual.add(RectF(left, top, right, bottom))
                                }
                                1 -> {
                                    ultimoColorDetectado = "Rojo"
                                    cintaEncontradaEnEsteFrame = true
                                }
                                2 -> {
                                    ultimoColorDetectado = "Café"
                                    cintaEncontradaEnEsteFrame = true
                                }
                                3 -> {
                                    ultimoColorDetectado = "Negra"
                                    cintaEncontradaEnEsteFrame = true
                                }
                            }
                        }
                    }

                    // Enviamos las cajas de las manos detectadas en este cuadro al Tracker
                    bananaTracker.updateTracks(cajasManosCuadroActual)

                    // Si hay IDs registrados en memoria, asumimos que hay mínimo un racimo enfocado
                    val racimoEnVista = if (bananaTracker.totalUniqueHandsCount > 0) 1 else 0

                    // Si en este frame ninguna cinta superó el 50%, no alteramos el 'ultimoColorDetectado'
                    // para mitigar parpadeos bruscos, a menos que el tracker esté completamente vacío.
                    if (!cintaEncontradaEnEsteFrame && bananaTracker.totalUniqueHandsCount == 0) {
                        ultimoColorDetectado = "Sin Cinta"
                    }

                    // Envío ordenado de resultados limpios al hilo principal de la interfaz
                    runOnUiThread {
                        tvTotalRacimos.text = "Total de Racimos Enfocados: $racimoEnVista"
                        tvTotalManos.text = "Número de Manos Contadas: ${bananaTracker.totalUniqueHandsCount}"
                        tvColorCinta.text = "Color de la Cinta: $ultimoColorDetectado"
                    }
                } catch (e: Exception) {
                    Log.e("TensorFlow", "Error en la inferencia del modelo", e)
                }
            }
        }
        imageProxy.close()
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * 640 * 640 * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(640 * 640)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until 640) {
            for (j in 0 until 640) {
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16) and 0xFF) / 255.0f)
                byteBuffer.putFloat(((value shr 8) and 0xFF) / 255.0f)
                byteBuffer.putFloat((value and 0xFF) / 255.0f)
            }
        }
        return byteBuffer
    }

    private fun mostrarDialogoExplicativo() {
        AlertDialog.Builder(this)
            .setTitle("Permiso de Cámara Requerido")
            .setMessage("Esta aplicación necesita acceder a la cámara para escanear las manos de banano en tiempo real.")
            .setPositiveButton("Conceder Permiso") { _, _ ->
                ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
            }
            .setNegativeButton("Salir") { _, _ -> finish() }
            .setCancelable(false)
            .show()
    }

    private fun mostrarDialogoAjustesSistema() {
        AlertDialog.Builder(this)
            .setTitle("Permiso denegado")
            .setMessage("Por favor ve a los ajustes de la aplicación y activa el permiso de Cámara manualmente.")
            .setPositiveButton("Ir a Ajustes") { _, _ ->
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                    data = Uri.fromParts("package", packageName, null)
                }
                startActivity(intent)
            }
            .setNegativeButton("Cancelar") { _, _ -> finish() }
            .setCancelable(false)
            .show()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(rc: Int, perms: Array<String>, results: IntArray) {
        super.onRequestPermissionsResult(rc, perms, results)
        if (rc == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                if (!ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
                    mostrarDialogoAjustesSistema()
                } else {
                    finish()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tfLiteInterpreter?.close()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}