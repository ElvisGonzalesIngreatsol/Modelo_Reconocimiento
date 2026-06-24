package com.example.reconocimiento_manos

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.provider.Settings
import android.util.Log
import android.graphics.Color
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
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
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var tvTotalRacimos: TextView
    private lateinit var tvTotalManos: TextView
    private lateinit var tvColorCinta: TextView
    private lateinit var btnResetTracker: FloatingActionButton
    private lateinit var btnUploadImage: FloatingActionButton

    private lateinit var cameraExecutor: ExecutorService
    private var tfLiteInterpreter: Interpreter? = null
    private val bananaTracker = BananaTracker()
    private var ultimoColorDetectado = "Sin Cinta"

    // Selector interactivo para abrir la galería del teléfono
    private val selectImageLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            val imageUri: Uri? = result.data?.data
            if (imageUri != null) {
                try {
                    val imageStream: InputStream? = contentResolver.openInputStream(imageUri)
                    val selectedBitmap = BitmapFactory.decodeStream(imageStream)
                    if (selectedBitmap != null) {
                        Toast.makeText(this, "Procesando imagen seleccionada...", Toast.LENGTH_SHORT).show()
                        procesarImagenEstatica(selectedBitmap)
                    }
                } catch (e: Exception) {
                    Log.e("Gallery", "Error al cargar imagen", e)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        overlayView = findViewById(R.id.overlayView)
        tvTotalRacimos = findViewById(R.id.tvTotalRacimos)
        tvTotalManos = findViewById(R.id.tvTotalManos)
        tvColorCinta = findViewById(R.id.tvColorCinta)
        btnResetTracker = findViewById(R.id.btnResetTracker)
        btnUploadImage = findViewById(R.id.btnUploadImage)

        btnResetTracker.setOnClickListener {
            bananaTracker.reset()

            ultimoColorDetectado = "Sin Cinta"
            overlayView.setResults(emptyList())
            tvTotalRacimos.text = "Total de Racimos Enfocados: 0"
            tvTotalManos.text = "Número de Manos Contadas: 0"
            tvColorCinta.text = "Color de la Cinta: Sin Cinta"
            Toast.makeText(this, "Contador reiniciado.", Toast.LENGTH_SHORT).show()
        }

        btnUploadImage.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            selectImageLauncher.launch(intent)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        initObjectDetector()

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
            val options = Interpreter.Options().apply { setNumThreads(4) }
            tfLiteInterpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            Log.e("TensorFlow", "Error cargando modelo", e)
        }
    }

    // ANALIZADOR DE VIDEO EN VIVO (CÁMARA)
    @androidx.camera.core.ExperimentalGetImage
    private fun procesarImagenConIA(imageProxy: ImageProxy) {
        val interpreter = tfLiteInterpreter
        if (interpreter != null) {
            val bitmap = imageProxyToBitmap(imageProxy)
            if (bitmap != null) {
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
                val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)
                val outputMap = Array(1) { Array(6) { FloatArray(8400) } }

                try {
                    interpreter.run(inputBuffer, outputMap)

                    val candidatasManos = mutableListOf<Pair<RectF, Float>>()
                    var mejorCintaDetectada: RectF? = null
                    var maxConfianzaCinta = 0.50f
                    val confUmbralMano = 0.55f // Umbral alto para evitar falsos positivos en vivo

                    // 1. Filtrado crudo inicial
                    for (i in 0 until 8400) {
                        val confCinta = outputMap[0][4][i]
                        val confMano = outputMap[0][5][i]

                        if (confMano > confUmbralMano && confMano > confCinta) {
                            val cx = outputMap[0][0][i]
                            val cy = outputMap[0][1][i]
                            val w = outputMap[0][2][i]
                            val h = outputMap[0][3][i]
                            candidatasManos.add(Pair(RectF(cx - w/2, cy - h/2, cx + w/2, cy + h/2), confMano))
                        } else if (confCinta > maxConfianzaCinta && confCinta > confMano) {
                            maxConfianzaCinta = confCinta
                            val cx = outputMap[0][0][i]
                            val cy = outputMap[0][1][i]
                            val w = outputMap[0][2][i]
                            val h = outputMap[0][3][i]
                            mejorCintaDetectada = RectF(cx - w/2, cy - h/2, cx + w/2, cy + h/2)
                        }
                    }

                    // 2. NMS ULTRA ESTRICTO EN VIVO (Elimina los cuadros duplicados en la misma mano)
                    val ordenadas = candidatasManos.sortedByDescending { it.second }.toMutableList()
                    val manosFiltradas = mutableListOf<RectF>()

                    while (ordenadas.isNotEmpty()) {
                        val primera = ordenadas.removeAt(0)
                        manosFiltradas.add(primera.first)

                        val iterator = ordenadas.iterator()
                        while (iterator.hasNext()) {
                            val siguiente = iterator.next()
                            // Si se superponen más del 30%, es la misma mano. Se elimina.
                            if (calculateIoU(primera.first, siguiente.first) > 0.30f) {
                                iterator.remove()
                            }
                        }
                    }

                    // 3. Preparar dibujo y tracking
                    val cajasParaDibujarOverlay = mutableListOf<BoxDibujo>()
                    val scaleX = viewFinder.width.toFloat() / 640f
                    val scaleY = viewFinder.height.toFloat() / 640f

                    for (rectMano in manosFiltradas) {
                        val screenRect = RectF(rectMano.left * scaleX, rectMano.top * scaleY, rectMano.right * scaleX, rectMano.bottom * scaleY)
                        cajasParaDibujarOverlay.add(BoxDibujo(screenRect, "Mano", true))
                    }

                    // Pasamos las cajas limpias al tracker (sin duplicados)
                    bananaTracker.updateTracks(manosFiltradas, 640f)

                    if (mejorCintaDetectada != null) {
                        ultimoColorDetectado = obtenerColorDeCinta(bitmap, mejorCintaDetectada)
                        val screenRectCinta = RectF(mejorCintaDetectada.left * scaleX, mejorCintaDetectada.top * scaleY, mejorCintaDetectada.right * scaleX, mejorCintaDetectada.bottom * scaleY)
                        cajasParaDibujarOverlay.add(BoxDibujo(screenRectCinta, "Cinta: $ultimoColorDetectado", false))
                    }

                    val racimoEnVista = if (manosFiltradas.isNotEmpty()) 1 else 0

                    runOnUiThread {
                        overlayView.setResults(cajasParaDibujarOverlay)
                        tvTotalRacimos.text = "Total de Racimos Enfocados: $racimoEnVista"

                        // Muestra cuántas manos hay en este instante frente a la cámara y cuántas van acumuladas en total
                        tvTotalManos.text = "Manos en vista: ${manosFiltradas.size} | Total Contadas: ${bananaTracker.totalUniqueHandsCount}"
                        tvColorCinta.text = "Color de la Cinta: $ultimoColorDetectado"
                    }
                } catch (e: Exception) {
                    Log.e("TensorFlow", "Error en inferencia de video", e)
                }
            }
        }
        imageProxy.close()
    }

    private fun procesarImagenEstatica(bitmapOriginal: Bitmap) {
        val interpreter = tfLiteInterpreter ?: return

        val resizedBitmap = Bitmap.createScaledBitmap(bitmapOriginal, 640, 640, true)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)
        val outputMap = Array(1) { Array(6) { FloatArray(8400) } }

        try {
            interpreter.run(inputBuffer, outputMap)

            val candidatasManos = mutableListOf<Pair<RectF, Float>>()
            var mejorCintaDetectada: RectF? = null
            var maxConfianzaCinta = 0.50f
            val confUmbralMano = 0.55f

            for (i in 0 until 8400) {
                val confCinta = outputMap[0][4][i]
                val confMano = outputMap[0][5][i]

                if (confMano > confUmbralMano && confMano > confCinta) {
                    val cx = outputMap[0][0][i]
                    val cy = outputMap[0][1][i]
                    val w = outputMap[0][2][i]
                    val h = outputMap[0][3][i]
                    candidatasManos.add(Pair(RectF(cx - w/2, cy - h/2, cx + w/2, cy + h/2), confMano))
                } else if (confCinta > maxConfianzaCinta && confCinta > confMano) {
                    maxConfianzaCinta = confCinta
                    val cx = outputMap[0][0][i]
                    val cy = outputMap[0][1][i]
                    val w = outputMap[0][2][i]
                    val h = outputMap[0][3][i]
                    mejorCintaDetectada = RectF(cx - w/2, cy - h/2, cx + w/2, cy + h/2)
                }
            }

            // Ejecutar NMS estricto para la foto de galería
            val ordenadas = candidatasManos.sortedByDescending { it.second }.toMutableList()
            val manosFiltradas = mutableListOf<RectF>()

            while (ordenadas.isNotEmpty()) {
                val primera = ordenadas.removeAt(0)
                manosFiltradas.add(primera.first)

                val iterator = ordenadas.iterator()
                while (iterator.hasNext()) {
                    val siguiente = iterator.next()
                    if (calculateIoU(primera.first, siguiente.first) > 0.25f) {
                        iterator.remove()
                    }
                }
            }

            val conteoManosFoto = manosFiltradas.size
            var colorCintaFoto = "Sin Cinta"

            // CREAR UNA COPIA MUTABLE DEL BITMAP ORIGINAL PARA PINTAR
            val bitmapResultadoConCuadros = bitmapOriginal.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = android.graphics.Canvas(bitmapResultadoConCuadros)

            val paintManoModal = android.graphics.Paint().apply {
                color = android.graphics.Color.parseColor("#9C27B0") // Morado para manos
                style = android.graphics.Paint.Style.STROKE
                strokeWidth = bitmapResultadoConCuadros.width * 0.007f
            }
            val paintCintaModal = android.graphics.Paint().apply {
                color = android.graphics.Color.parseColor("#4CAF50") // Verde para la cinta
                style = android.graphics.Paint.Style.STROKE
                strokeWidth = bitmapResultadoConCuadros.width * 0.009f
            }

            // Dibujar las manos mapeadas a la escala original de la foto
            for (rectMano in manosFiltradas) {
                val realRect = RectF(
                    rectMano.left * bitmapResultadoConCuadros.width / 640f,
                    rectMano.top * bitmapResultadoConCuadros.height / 640f,
                    rectMano.right * bitmapResultadoConCuadros.width / 640f,
                    rectMano.bottom * bitmapResultadoConCuadros.height / 640f
                )
                canvas.drawRect(realRect, paintManoModal)
            }

            // Dibujar la cinta si existe
            if (mejorCintaDetectada != null) {
                colorCintaFoto = obtenerColorDeCinta(bitmapOriginal, mejorCintaDetectada)
                val realRectCinta = RectF(
                    mejorCintaDetectada.left * bitmapResultadoConCuadros.width / 640f,
                    mejorCintaDetectada.top * bitmapResultadoConCuadros.height / 640f,
                    mejorCintaDetectada.right * bitmapResultadoConCuadros.width / 640f,
                    mejorCintaDetectada.bottom * bitmapResultadoConCuadros.height / 640f
                )
                canvas.drawRect(realRectCinta, paintCintaModal)
            }

            val racimoDetectado = if (conteoManosFoto > 0) 1 else 0

            runOnUiThread {
                // Se envía el bitmap ya modificado con los canvas aplicados
                mostrarModalResultados(bitmapResultadoConCuadros, racimoDetectado, conteoManosFoto, colorCintaFoto)
            }

        } catch (e: Exception) {
            Log.e("TensorFlow", "Error procesando imagen estática", e)
        }
    }

    // Función auxiliar de soporte para el cálculo de intersección (IoU) si se requiere de forma local
    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectionLeft = maxOf(box1.left, box2.left)
        val intersectionTop = maxOf(box1.top, box2.top)
        val intersectionRight = minOf(box1.right, box2.right)
        val intersectionBottom = minOf(box1.bottom, box2.bottom)

        if (intersectionLeft < intersectionRight && intersectionTop < intersectionBottom) {
            val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
            val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
            val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
            return intersectionArea / (box1Area + box2Area - intersectionArea)
        }
        return 0f
    }

    // 2. NUEVA FUNCIÓN: Genera y despliega el Modal Emergente con el diseño y resultados
    private fun mostrarModalResultados(bitmapResultado: Bitmap, racimos: Int, manos: Int, colorCinta: String) {
        val builder = AlertDialog.Builder(this)

        val layoutModal = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(40, 40, 40, 40)
        }

        val imageView = android.widget.ImageView(this).apply {
            layoutParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                800
            )
            setImageBitmap(bitmapResultado)
            scaleType = android.widget.ImageView.ScaleType.FIT_CENTER
        }

        val tvInfo = TextView(this).apply {
            text = """
            Resultados del Análisis:
            -------------------------------------------
            • Racimos Identificados: $racimos
            • Número de Manos Únicas: $manos
            • Color de Cinta Encontrado: $colorCinta
        """.trimIndent()

            // CORRECCIÓN AQUÍ: Se asigna el tamaño como Float directamente y se especifica la unidad SP correctamente
            setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 16f)

            setTextColor(android.graphics.Color.BLACK)
            setPadding(0, 30, 0, 10)
        }

        layoutModal.addView(imageView)
        layoutModal.addView(tvInfo)

        builder.setView(layoutModal)
        builder.setTitle("Análisis de Imagen Completado")
        builder.setPositiveButton("Aceptar") { dialog, _ -> dialog.dismiss() }

        val dialog = builder.create()
        dialog.show()
    }

    private fun ejecutarInferenciaYOLO(bitmap: Bitmap) {
        val interpreter = tfLiteInterpreter ?: return
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)
        val outputMap = Array(1) { Array(6) { FloatArray(8400) } }

        try {
            interpreter.run(inputBuffer, outputMap)

            val candidatasManos = mutableListOf<Pair<RectF, Float>>()
            var mejorCintaDetectada: RectF? = null
            var maxConfianzaCinta = 0.50f
            val confUmbralMano = 0.45f

            for (i in 0 until 8400) {
                val confCinta = outputMap[0][4][i]
                val confMano = outputMap[0][5][i]

                if (confMano > confUmbralMano && confMano > confCinta) {
                    val cx = outputMap[0][0][i]
                    val cy = outputMap[0][1][i]
                    val w = outputMap[0][2][i]
                    val h = outputMap[0][3][i]
                    candidatasManos.add(Pair(RectF(cx - w/2, cy - h/2, cx + w/2, cy + h/2), confMano))
                } else if (confCinta > maxConfianzaCinta && confCinta > confMano) {
                    maxConfianzaCinta = confCinta
                    val cx = outputMap[0][0][i]
                    val cy = outputMap[0][1][i]
                    val w = outputMap[0][2][i]
                    val h = outputMap[0][3][i]
                    mejorCintaDetectada = RectF(cx - w/2, cy - h/2, cx + w/2, cy + h/2)
                }
            }

            val manosFiltradas = aplicarNMS(candidatasManos, iouThreshold = 0.40f)
            val cajasManosCuadroActual = mutableListOf<RectF>()
            val cajasParaDibujarOverlay = mutableListOf<BoxDibujo>()

            val scaleX = viewFinder.width.toFloat() / 640f
            val scaleY = viewFinder.height.toFloat() / 640f

            for (rectMano in manosFiltradas) {
                cajasManosCuadroActual.add(rectMano)
                val screenRect = RectF(rectMano.left * scaleX, rectMano.top * scaleY, rectMano.right * scaleX, rectMano.bottom * scaleY)
                cajasParaDibujarOverlay.add(BoxDibujo(screenRect, "Mano", true))
            }

            bananaTracker.updateTracks(cajasManosCuadroActual, 640f)

            if (mejorCintaDetectada != null) {
                ultimoColorDetectado = obtenerColorDeCinta(bitmap, mejorCintaDetectada)
                val screenRectCinta = RectF(mejorCintaDetectada.left * scaleX, mejorCintaDetectada.top * scaleY, mejorCintaDetectada.right * scaleX, mejorCintaDetectada.bottom * scaleY)
                cajasParaDibujarOverlay.add(BoxDibujo(screenRectCinta, "Cinta: $ultimoColorDetectado", false))
            }

            val racimoEnVista = if (bananaTracker.totalUniqueHandsCount > 0) 1 else 0

            runOnUiThread {
                // Notifica y refresca la pantalla transparente con las líneas al instante
                overlayView.setResults(cajasParaDibujarOverlay)

                tvTotalRacimos.text = "Total de Racimos Enfocados: $racimoEnVista"
                tvTotalManos.text = "Número de Manos Contadas: ${bananaTracker.totalUniqueHandsCount}"
                tvColorCinta.text = "Color de la Cinta: $ultimoColorDetectado"
            }
        } catch (e: Exception) {
            Log.e("TensorFlow", "Error en inferencia", e)
        }
    }

    private fun aplicarNMS(cajas: List<Pair<RectF, Float>>, iouThreshold: Float): List<RectF> {
        val ordenadas = cajas.sortedByDescending { it.second }.toMutableList()
        val resultado = mutableListOf<RectF>()
        while (ordenadas.isNotEmpty()) {
            val primera = ordenadas.removeAt(0)
            resultado.add(primera.first)
            val iterator = ordenadas.iterator()
            while (iterator.hasNext()) {
                val siguiente = iterator.next()
                if (calcularIoUEnMainActivity(primera.first, siguiente.first) > iouThreshold) {
                    iterator.remove()
                }
            }
        }
        return resultado
    }

    private fun calcularIoUEnMainActivity(box1: RectF, box2: RectF): Float {
        val x1 = maxOf(box1.left, box2.left)
        val y1 = maxOf(box1.top, box2.top)
        val x2 = minOf(box1.right, box2.right)
        val y2 = minOf(box1.bottom, box2.bottom)
        if (x1 < x2 && y1 < y2) {
            val intersection = (x2 - x1) * (y2 - y1)
            val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
            val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
            return intersection / (box1Area + box2Area - intersection)
        }
        return 0f
    }

    private fun checkCameraPermission() {
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(viewFinder.surfaceProvider) }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor) { img -> procesarImagenConIA(img) } }
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
            } catch (e: Exception) {
                Log.e("CameraX", "Error de inicio", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val planes = imageProxy.planes
        val buffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * imageProxy.width
        val bitmap = Bitmap.createBitmap(imageProxy.width + rowPadding / pixelStride, imageProxy.height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)
        if (imageProxy.imageInfo.rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()) }
            return Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height, matrix, true)
        }
        return bitmap
    }

    private fun obtenerColorDeCinta(bitmapOriginal: Bitmap, rectCinta: RectF): String {
        val left = maxOf(0, rectCinta.left.toInt())
        val top = maxOf(0, rectCinta.top.toInt())
        val width = minOf(bitmapOriginal.width - left, rectCinta.width().toInt())
        val height = minOf(bitmapOriginal.height - top, rectCinta.height().toInt())
        if (width <= 0 || height <= 0) return "Detectando..."
        val cropCinta = Bitmap.createBitmap(bitmapOriginal, left, top, width, height)
        var sumHue = 0f; var sumSat = 0f; val totalPixeles = cropCinta.width * cropCinta.height
        val hsv = FloatArray(3)
        for (x in 0 until cropCinta.width) {
            for (y in 0 until cropCinta.height) {
                Color.colorToHSV(cropCinta.getPixel(x, y), hsv)
                sumHue += hsv[0]; sumSat += hsv[1]
            }
        }
        val avgHue = sumHue / totalPixeles
        return when (avgHue) {
            in 0.0..20.0 -> "Rojo"
            in 20.0..50.0 -> "Naranja"
            in 50.0..85.0 -> "Amarillo"
            in 85.0..160.0 -> "Verde"
            else -> "Azul"
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * 640 * 640 * 3 * 4).apply { order(ByteOrder.nativeOrder()) }
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

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED }
    override fun onDestroy() { super.onDestroy(); tfLiteInterpreter?.close(); cameraExecutor.shutdown() }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}