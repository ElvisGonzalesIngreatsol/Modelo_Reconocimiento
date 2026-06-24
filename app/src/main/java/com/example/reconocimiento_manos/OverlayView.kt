package com.example.reconocimiento_manos

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

data class BoxDibujo(val rect: RectF, val label: String, val esMano: Boolean)

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private val cajasParaDibujar = mutableListOf<BoxDibujo>()

    private val paintMano = Paint().apply {
        color = Color.parseColor("#9C27B0") // Morado
        style = Paint.Style.STROKE
        strokeWidth = 8f // Línea gruesa visible
        isAntiAlias = true
    }

    private val paintCinta = Paint().apply {
        color = Color.parseColor("#4CAF50") // Verde
        style = Paint.Style.STROKE
        strokeWidth = 9f
        isAntiAlias = true
    }

    private val paintTexto = Paint().apply {
        color = Color.WHITE
        textSize = 34f
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val paintFondoTexto = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    fun setResults(nuevasCajas: List<BoxDibujo>) {
        cajasParaDibujar.clear()
        cajasParaDibujar.addAll(nuevasCajas)
        invalidate() // <-- IMPORTANTE: Obliga a la pantalla a redibujarse al instante
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (box in cajasParaDibujar) {
            val pincelCaja = if (box.esMano) paintMano else paintCinta
            canvas.drawRect(box.rect, pincelCaja)

            // Fondo de la etiqueta de texto
            paintFondoTexto.color = pincelCaja.color
            canvas.drawRect(
                box.rect.left,
                box.rect.top - 45f,
                box.rect.left + pincelCaja.measureText(box.label) + 30f,
                box.rect.top,
                paintFondoTexto
            )
            canvas.drawText(box.label, box.rect.left + 10f, box.rect.top - 12f, paintTexto)
        }
    }
}