// Local, image-only OCR proposals. No instruction or target annotation enters Vision.
import Foundation
import ImageIO
import Vision

func pixelBox(_ rectangle: CGRect, _ width: Int, _ height: Int, _ x: Int, _ y: Int) -> [Double] {
    return [Double(x) + Double(rectangle.minX) * Double(width),
            Double(y) + (1 - Double(rectangle.maxY)) * Double(height),
            Double(x) + Double(rectangle.maxX) * Double(width),
            Double(y) + (1 - Double(rectangle.minY)) * Double(height)]
}

func origins(_ length: Int, _ tile: Int) -> [Int] {
    if tile == 0 || length <= tile { return [0] }
    var result = [0]
    while result.last! + tile < length {
        result.append(min(result.last! + tile - 128, length - tile))
    }
    return result
}

do {
    guard CommandLine.arguments.count == 2 || CommandLine.arguments.count == 3 else {
        throw NSError(domain: "screen_ocr", code: 1, userInfo: [NSLocalizedDescriptionKey: "Supply one image path"])
    }
    let tileSize = CommandLine.arguments.count == 3 ? Int(CommandLine.arguments[2]) ?? -1 : 0
    guard tileSize == 0 || tileSize == 1024 else {
        throw NSError(domain: "screen_ocr", code: 3, userInfo: [NSLocalizedDescriptionKey: "Tile size must be 0 or 1024"])
    }
    let start = Date()
    let url = URL(fileURLWithPath: CommandLine.arguments[1])
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "screen_ocr", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot decode image"])
    }
    let request = VNRecognizeTextRequest()
    request.revision = VNRecognizeTextRequestRevision3
    request.recognitionLevel = .fast
    request.recognitionLanguages = ["en-US"]
    request.usesLanguageCorrection = false
    request.minimumTextHeight = 0
    // Kept explicit for compatibility with the published VNRecognizeTextRequest API.
    request.usesCPUOnly = true
    let expression = try NSRegularExpression(pattern: "\\S+")
    var proposals: [[String: Any]] = []
    var tileCount = 0
    for y in origins(image.height, tileSize) {
      for x in origins(image.width, tileSize) {
        let width = tileSize == 0 ? image.width : min(tileSize, image.width)
        let height = tileSize == 0 ? image.height : min(tileSize, image.height)
        guard let crop = image.cropping(to: CGRect(x: x, y: y, width: width, height: height)) else { continue }
        try VNImageRequestHandler(cgImage: crop, orientation: .up, options: [:]).perform([request])
        tileCount += 1
        for observation in request.results ?? [] {
        guard let candidate = observation.topCandidates(1).first else { continue }
        let text = candidate.string
        proposals.append(["text": text, "bbox_xyxy": pixelBox(observation.boundingBox, width, height, x, y),
                          "kind": "line", "confidence": candidate.confidence])
        for match in expression.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let range = Range(match.range, in: text),
                  let rectangle = try candidate.boundingBox(for: range) else { continue }
            proposals.append(["text": String(text[range]), "bbox_xyxy": pixelBox(rectangle.boundingBox, width, height, x, y),
                              "kind": "word", "confidence": candidate.confidence])
        }
        }
      }
    }
    let result: [String: Any] = ["width": image.width, "height": image.height, "proposals": proposals,
        "ocr_seconds": Date().timeIntervalSince(start), "revision": request.revision,
        "recognition_level": "fast", "language": "en-US", "language_correction": false, "cpu_only_requested": true,
        "tile_size": tileSize, "tile_overlap": tileSize == 0 ? 0 : 128, "tile_count": tileCount]
    let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
} catch {
    FileHandle.standardError.write(Data(("OCR failed: \(error.localizedDescription)\n").utf8))
    exit(1)
}
