import Foundation
import Testing
@testable import TransiroYT

@Test func copiedAPIConnectionNeverCopiesCredentialsOrAssignment() {
    let original = APIPreset(name: "Example", enabled: true, model: "model", apiKey: "dummy", hasKey: true)
    let copy = original.duplicate()
    #expect(copy.id != original.id)
    #expect(copy.apiKey == nil && !copy.hasKey && !copy.enabled)
    #expect(copy.baseUrl == original.baseUrl && copy.model == original.model)
}

@Test func credentialScopeIncludesEveryDestinationAndNormalizesDefaultPorts() {
    let original = APIPreset()
    var draft = original
    draft.inferenceUrl = "https://api.openai.com:443/custom"
    #expect(draft.credentialScope == original.credentialScope)
    draft.modelsUrl = "https://different.example/models"
    #expect(draft.credentialScope != original.credentialScope)
    draft = original
    draft.protocol = "responses"
    #expect(draft.credentialScope != original.credentialScope)
}

@Test func simpleModeBecomesTheOrderTheBackendUses() {
    var config = TaskRoutingSettings()
    config.presets = [APIPreset(id: String(repeating: "a", count: 32), enabled: true, model: "m"),
                      APIPreset(id: String(repeating: "b", count: 32), enabled: false, model: "m")]
    config.engines.bing = true
    config.engines.gemma = false
    config.useExplicitOrder()
    let api = "api:" + String(repeating: "a", count: 32)
    #expect(config.enabled)
    #expect(config.translation.chain == ["plamo", api, "bing"])
    #expect(config.language.chain == [api])
    #expect(config.shortening.chain == [api])
}

@Test func usingAnAIAddsItLastAndTasksOnlyOfferWhatTheBackendAccepts() {
    var config = TaskRoutingSettings(enabled: true)
    config.setUse("gemma", for: .translation, true)
    #expect(config.translation.chain == ["plamo", "gemma"])
    config.setUse("plamo", for: .translation, false)
    config.setUse("plamo", for: .translation, true)
    #expect(config.translation.chain == ["gemma", "plamo"])
    #expect(!config.candidates(.language).contains("plamo") && !config.candidates(.shortening).contains("bing"))
    config.language.chain = []
    #expect(config.problem != nil)
}
