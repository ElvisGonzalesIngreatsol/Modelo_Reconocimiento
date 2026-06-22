package com.example.reconocimiento_manos

import android.graphics.RectF

data class TrackedHand(
    val id: Int,
    var boundingBox: RectF,
    var missedFrames: Int = 0 // Contador de fotogramas que ha estado desaparecida
)

class BananaTracker {
    private var nextId = 1
    private val activeTracks = mutableListOf<TrackedHand>()

    var totalUniqueHandsCount = 0
        private set

    private val countedIds = mutableSetOf<Int>()

    // CONFIGURACIÓN DE AJUSTE (HIERPARÁMETROS DEL TRACKER)
    private val IOU_THRESHOLD = 0.60f    // Sube al 60% para evitar que el parpadeo genere IDs falsos
    private val MAX_MISSED_FRAMES = 15   // Si la mano parpadea o se tapa, la recordamos por 15 frames antes de borrarla

    fun updateTracks(detectedBoxes: List<RectF>) {
        val updatedTracks = mutableListOf<TrackedHand>()
        val usedBoxes = BooleanArray(detectedBoxes.size)

        // 1. Intentar emparejar las detecciones nuevas con los IDs existentes en memoria
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
                // La mano sigue en pantalla: actualizamos coordenadas y reiniciamos su contador de fallos
                track.boundingBox = detectedBoxes[bestBoxIdx]
                track.missedFrames = 0
                usedBoxes[bestBoxIdx] = true
                updatedTracks.add(track)
            } else {
                // La mano no se detectó en este fotograma. Le sumamos un fallo.
                track.missedFrames++
                // Si aún no supera el límite de tolerancia, la retenemos para que no pierda su ID
                if (track.missedFrames <= MAX_MISSED_FRAMES) {
                    updatedTracks.add(track)
                }
            }
        }

        // 2. Registrar manos completamente nuevas que acaban de entrar a la toma
        for (i in detectedBoxes.indices) {
            if (!usedBoxes[i]) {
                val newBox = detectedBoxes[i]

                // TRUCO DE CONTROL: Para evitar que el ruido de los bordes sume infinitamente,
                // solo registramos la mano si está dentro de la zona central de la pantalla (Margen del 5%)
                if (newBox.left > 5f && newBox.right < 635f) {
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

    fun resetTracker() {
        nextId = 1
        activeTracks.clear()
        countedIds.clear()
        totalUniqueHandsCount = 0
    }
}