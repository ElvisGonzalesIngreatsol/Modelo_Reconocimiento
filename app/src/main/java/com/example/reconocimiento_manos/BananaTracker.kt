package com.example.reconocimiento_manos

import android.graphics.RectF

data class TrackedHand(
    val id: Int,
    var boundingBox: RectF,
    var missedFrames: Int = 0
)

class BananaTracker {
    private var nextId = 1
    private val activeTracks = mutableListOf<TrackedHand>()
    var totalUniqueHandsCount = 0
        private set

    private val countedIds = mutableSetOf<Int>()

    fun updateTracks(detectedBoxes: List<RectF>) {
        val updatedTracks = mutableListOf<TrackedHand>()
        val usedBoxes = BooleanArray(detectedBoxes.size)

        // 1. Intentar emparejar las detecciones nuevas con los IDs existentes (IoU)
        for (track in activeTracks) {
            var bestBoxIdx = -1
            var bestIoU = 0.4f // Umbral de coincidencia del 40%

            for (i in detectedBoxes.indices) {
                if (usedBoxes[i]) continue
                val iou = calculateIoU(track.boundingBox, detectedBoxes[i])
                if (iou > bestIoU) {
                    bestIoU = iou
                    bestBoxIdx = i
                }
            }

            if (bestBoxIdx != -1) {
                // La mano sigue en pantalla: actualizamos coordenadas e ID
                track.boundingBox = detectedBoxes[bestBoxIdx]
                track.missedFrames = 0
                updatedTracks.add(track)
                usedBoxes[bestBoxIdx] = true
            } else {
                // Si la mano desaparece un frame (pestañeo), le damos un margen de 5 cuadros
                track.missedFrames++
                if (track.missedFrames < 5) {
                    updatedTracks.add(track)
                }
            }
        }

        // 2. Registrar manos nuevas encontradas al girar el racimo
        for (i in detectedBoxes.indices) {
            if (!usedBoxes[i]) {
                val newTrack = TrackedHand(id = nextId++, boundingBox = detectedBoxes[i])
                updatedTracks.add(newTrack)

                if (!countedIds.contains(newTrack.id)) {
                    countedIds.add(newTrack.id)
                    totalUniqueHandsCount++
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
        return 0.0f
    }

    fun resetTracker() {
        nextId = 1
        activeTracks.clear()
        countedIds.clear()
        totalUniqueHandsCount = 0
    }
}