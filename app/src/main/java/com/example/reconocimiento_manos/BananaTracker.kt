package com.example.reconocimiento_manos

import android.graphics.RectF

data class TrackedHand(
    val id: Int,
    var boundingBox: RectF,
    var missedFrames: Int = 0 // Cuántos fotogramas lleva sin detectarse
)

class BananaTracker {
    private var nextId = 1
    private val activeTracks = mutableListOf<TrackedHand>()

    var totalUniqueHandsCount = 0
        private set

    private val countedIds = mutableSetOf<Int>()

    // PARÁMETROS AJUSTADOS PARA EVITAR CONTAR MANOS REPETIDAS
    private val IOU_THRESHOLD = 0.15f     // Más bajo para que asocie la mano aunque se mueva rápido
    private val MAX_MISSED_FRAMES = 35    // Memoria alta (35 frames): si el modelo parpadea, mantiene el ID viejo

    fun updateTracks(detectedBoxes: List<RectF>, frameWidth: Float) {
        val updatedTracks = mutableListOf<TrackedHand>()
        val usedBoxes = BooleanArray(detectedBoxes.size)

        // 1. Intentar emparejar las manos en movimiento con la memoria existente
        for (track in activeTracks) {
            var bestBoxIdx = -1
            var bestIoU = IOU_THRESHOLD

            for (i in detectedBoxes.indices) {
                if (usedBoxes[i]) continue
                val iou = calculateIoU(track.boundingBox, detectedBoxes[i])
                if (iou > bestIoU) {
                    bestIoU = iou
                    bestBoxIdx = i
                }
            }

            if (bestBoxIdx != -1) {
                // La mano sigue siendo la misma: actualizamos su posición y reiniciamos el contador de olvido
                track.boundingBox = detectedBoxes[bestBoxIdx]
                track.missedFrames = 0
                usedBoxes[bestBoxIdx] = true
                updatedTracks.add(track)
            } else {
                // No se detectó en este cuadro, le sumamos un cuadro de olvido
                track.missedFrames++
                if (track.missedFrames <= MAX_MISSED_FRAMES) {
                    updatedTracks.add(track) // La mantenemos viva en memoria
                }
            }
        }

        // 2. Registrar manos completamente nuevas (solo si no se emparejaron con nada previo)
        for (i in detectedBoxes.indices) {
            if (!usedBoxes[i]) {
                val newBox = detectedBoxes[i]

                // Evitamos registrar ruidos falsos en los bordes extremos de la pantalla
                if (newBox.left > 5f && newBox.right < (frameWidth - 5f)) {
                    val newTrack = TrackedHand(id = nextId++, boundingBox = newBox)
                    updatedTracks.add(newTrack)

                    if (!countedIds.contains(newTrack.id)) {
                        countedIds.add(newTrack.id)
                        totalUniqueHandsCount++
                    }
                }
            }
        }

        activeTracks.clear()
        activeTracks.addAll(updatedTracks)
    }

    fun reset() {
        nextId = 1
        activeTracks.clear()
        countedIds.clear()
        totalUniqueHandsCount = 0
    }

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
}