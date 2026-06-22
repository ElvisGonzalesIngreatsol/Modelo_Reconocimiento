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
            Log.d("TensorFlow", "¡Modelo YOLOv11 cargado exitosamente mediante Interpreter!")

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
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // Forzamos RGBA para manipulación directa de píxeles
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
            // Convertimos el ImageProxy a Bitmap nativo compatible
            val bitmap = imageProxyToBitmap(imageProxy)

            if (bitmap != null) {
                // Redimensionamos al tamaño de entrada que espera el modelo (640x640)
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
                val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

                // CONTENEDOR CORRECTO PARA LA SALIDA DE YOLOV11 SIN NMS: [1][6][8400]
                // Índices: 0,1,2,3 -> Cajas | 4 -> Confianza Clase 'cinta' | 5 -> Confianza Clase 'mano'
                val outputMap = Array(1) { Array(6) { FloatArray(8400) } }

                try {
                    interpreter.run(inputBuffer, outputMap)

                    // Lista temporal donde guardaremos las coordenadas de las manos vistas en ESTE cuadro físico
                    val cajasManosCuadroActual = mutableListOf<RectF>()
                    var mejorCintaDetectada: RectF? = null
                    var maxConfianzaCinta = 0.50f // Umbral del 50% para la cinta
                    val confUmbralMano = 0.50f    // Umbral del 50% para las manos

                    // Recorremos las 8400 predicciones brutas de la arquitectura YOLOv11
                    for (i in 0 until 8400) {
                        val confCinta = outputMap[0][4][i]
                        val confMano = outputMap[0][5][i]

                        // Si la probabilidad más alta pertenece a una MANO
                        if (confMano > confUmbralMano && confMano > confCinta) {
                            val cx = outputMap[0][0][i] * bitmap.width / 640f
                            val cy = outputMap[0][1][i] * bitmap.height / 640f
                            val w = outputMap[0][2][i] * bitmap.width / 640f
                            val h = outputMap[0][3][i] * bitmap.height / 640f

                            val rect = RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                            cajasManosCuadroActual.add(rect)
                        }
                        // Si la probabilidad más alta pertenece a una CINTA
                        else if (confCinta > maxConfianzaCinta && confCinta > confMano) {
                            maxConfianzaCinta = confCinta
                            val cx = outputMap[0][0][i] * bitmap.width / 640f
                            val cy = outputMap[0][1][i] * bitmap.height / 640f
                            val w = outputMap[0][2][i] * bitmap.width / 640f
                            val h = outputMap[0][3][i] * bitmap.height / 640f

                            mejorCintaDetectada = RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                        }
                    }

                    // Enviamos las cajas de las manos detectadas en este cuadro al Tracker por IoU
                    bananaTracker.updateTracks(cajasManosCuadroActual)

                    // Si el modelo localizó una cinta válida, extraemos dinámicamente su color real en HSV
                    if (mejorCintaDetectada != null) {
                        ultimoColorDetectado = obtenerColorDeCinta(bitmap, mejorCintaDetectada)
                    } else if (bananaTracker.totalUniqueHandsCount == 0) {
                        ultimoColorDetectado = "Sin Cinta"
                    }

                    // Si hay IDs registrados en memoria, asumimos que hay mínimo un racimo enfocado
                    val racimoEnVista = if (bananaTracker.totalUniqueHandsCount > 0) 1 else 0

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

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val planes = imageProxy.planes
        val buffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * imageProxy.width

        val bitmap = Bitmap.createBitmap(
            imageProxy.width + rowPadding / pixelStride,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(buffer)

        // Corregimos la rotación física del sensor del teléfono para procesar derecho el racimo
        if (imageProxy.imageInfo.rotationDegrees != 0) {
            val matrix = Matrix()
            matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            return Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height, matrix, true)
        }
        return bitmap
    }

    private fun obtenerColorDeCinta(bitmapOriginal: Bitmap, rectCinta: RectF): String {
        // Asegurar que el cuadro recortado no se salga de los límites reales de la foto capturada
        val left = maxOf(0, rectCinta.left.toInt())
        val top = maxOf(0, rectCinta.top.toInt())
        val width = minOf(bitmapOriginal.width - left, rectCinta.width().toInt())
        val height = minOf(bitmapOriginal.height - top, rectCinta.height().toInt())

        if (width <= 0 || height <= 0) return "Detectando..."

        // Recortamos la sub-imagen exacta donde la IA dice que está la cinta de edad
        val cropCinta = Bitmap.createBitmap(bitmapOriginal, left, top, width, height)

        var sumHue = 0f
        var sumSat = 0f
        var sumVal = 0f
        val totalPixeles = cropCinta.width * cropCinta.height
        val hsv = FloatArray(3)

        // Recorremos los píxeles para promediar el color real tapando sombras del cobertizo
        for (x in 0 until cropCinta.width) {
            for (y in 0 until cropCinta.height) {
                val pixel = cropCinta.getPixel(x, y)
                android.graphics.Color.colorToHSV(pixel, hsv)
                sumHue += hsv[0]
                sumSat += hsv[1]
                sumVal += hsv[2]
            }
        }

        val avgHue = sumHue / totalPixeles
        val avgSat = sumSat / totalPixeles
        val avgVal = sumVal / totalPixeles

        // Si está muy oscuro o muy opaco debido al entorno, se marca como sombra
        if (avgVal < 0.2f || avgSat < 0.15f) return "Buscando... (Sombra)"

        // Clasificación matemática según los grados del espectro HUE (0° a 360°)
        return when (avgHue) {
            in 0.0..20.0 -> "Rojo (Semana 1)"
            in 20.0..50.0 -> "Naranja (Semana 2)"
            in 50.0..85.0 -> "Amarillo / Verde Claro (Semana 3)"
            in 85.0..160.0 -> "Verde (Semana 4)"
            in 160.0..240.0 -> "Azul / Morado (Semana 5)"
            in 240.0..330.0 -> "Morado / Violeta (Semana 6)"
            else -> "Sin Cinta"
        }
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