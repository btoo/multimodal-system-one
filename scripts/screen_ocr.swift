// Local, image-only OCR proposals. No instruction or target annotation enters Vision.
import Foundation
import ImageIO
import Vision

func pixelBox(_ rectangle: CGRect, _ width: Int, _ height: Int) -> [Double] {
    return [Double(rectangle.minX) * Double(width),
            (1 - Double(rectangle.maxY)) * Double(height),
            Double(rectangle.maxX) * Double(width),
            (1 - Double(rectangle.minY)) * Double(height)]
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw NSError(domain: "screen_ocr", code: 1, userInfo: [NSLocalizedDescriptionKey: "Supply one image path"])
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
    try VNImageRequestHandler(cgImage: image, orientation: .up, options: [:]).perform([request])
    let expression = try NSRegularExpression(pattern: "\\S+")
    var proposals: [[String: Any]] = []
    for observation in request.results ?? [] {
        guard let candidate = observation.topCandidates(1).first else { continue }
        let text = candidate.string
        proposals.append(["text": text, "bbox_xyxy": pixelBox(observation.boundingBox, image.width, image.height),
                          "kind": "line", "confidence": candidate.confidence])
        for match in expression.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let range = Range(match.range, in: text),
                  let rectangle = try candidate.boundingBox(for: range) else { continue }
            proposals.append(["text": String(text[range]), "bbox_xyxy": pixelBox(rectangle.boundingBox, image.width, image.height),
                              "kind": "word", "confidence": candidate.confidence])
        }
    }
    let result: [String: Any] = ["width": image.width, "height": image.height, "proposals": proposals,
        "ocr_seconds": Date().timeIntervalSince(start), "revision": request.revision,
        "recognition_level": "fast", "language": "en-US", "language_correction": false, "cpu_only_requested": true]
    let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
} catch {
    FileHandle.standardError.write(Data(("OCR failed: \(error.localizedDescription)\n").utf8))
    exit(1)
}
