import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import { getHermesConfigRecord, saveHermesConfig } from '@/hermes'

import {
  $voiceInputMode,
  $voiceStopPhrase,
  applyVoiceRealtimeFromConfig,
  applyVoiceStopPhraseFromConfig,
  setVoiceInputMode
} from './voice-prefs'

const getHermesConfigRecordMock = vi.mocked(getHermesConfigRecord)
const saveHermesConfigMock = vi.mocked(saveHermesConfig)

beforeEach(() => {
  vi.clearAllMocks()
  $voiceInputMode.set('legacy')
  $voiceStopPhrase.set('stop')
  getHermesConfigRecordMock.mockResolvedValue({})
  saveHermesConfigMock.mockResolvedValue({ ok: true })
})

describe('applyVoiceStopPhraseFromConfig', () => {
  it('defaults to "stop" when the key is absent (backend default applies)', () => {
    applyVoiceStopPhraseFromConfig({ voice: {} })
    expect($voiceStopPhrase.get()).toBe('stop')

    applyVoiceStopPhraseFromConfig(null)
    expect($voiceStopPhrase.get()).toBe('stop')
  })

  it('uses the first configured phrase so a custom phrase renders correctly', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['goodbye hermes', 'stop'] } })
    expect($voiceStopPhrase.get()).toBe('goodbye hermes')
  })

  it('coerces a bare string like the backend does', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: 'halt' } })
    expect($voiceStopPhrase.get()).toBe('halt')
  })

  it('null phrase when stop phrases are disabled — no notice is shown', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: [] } })
    expect($voiceStopPhrase.get()).toBeNull()
  })

  it('malformed entries are skipped; all-blank list disables', () => {
    applyVoiceStopPhraseFromConfig({ voice: { stop_phrases: ['  ', ''] } })
    expect($voiceStopPhrase.get()).toBeNull()
  })
})

describe('applyVoiceRealtimeFromConfig', () => {
  it('does not hydrate realtime mode from a stale config whose broker gate is disabled', () => {
    applyVoiceRealtimeFromConfig({
      voice: {
        input_mode: 'realtime',
        realtime: {
          enabled: false,
          stt_provider: 'elevenlabs'
        }
      }
    })

    expect($voiceInputMode.get()).toBe('legacy')
  })
})

describe('setVoiceInputMode', () => {
  it('enables realtime with both the mode and backend broker gate in one config write', async () => {
    getHermesConfigRecordMock.mockResolvedValue({
      voice: {
        auto_tts: true,
        realtime: {
          enabled: false,
          stt_provider: 'elevenlabs'
        }
      }
    })

    await setVoiceInputMode('realtime')

    expect($voiceInputMode.get()).toBe('realtime')
    expect(saveHermesConfigMock).toHaveBeenCalledTimes(1)
    expect(saveHermesConfigMock).toHaveBeenCalledWith({
      voice: {
        auto_tts: true,
        input_mode: 'realtime',
        realtime: {
          enabled: true,
          stt_provider: 'elevenlabs'
        }
      }
    })
  })

  it('returns to legacy by disabling realtime without dropping existing realtime preferences', async () => {
    $voiceInputMode.set('realtime')
    getHermesConfigRecordMock.mockResolvedValue({
      voice: {
        input_mode: 'realtime',
        realtime: {
          enabled: true,
          stt_provider: 'openai'
        },
        stop_phrases: ['enough']
      }
    })

    await setVoiceInputMode('legacy')

    expect($voiceInputMode.get()).toBe('legacy')
    expect(saveHermesConfigMock).toHaveBeenCalledTimes(1)
    expect(saveHermesConfigMock).toHaveBeenCalledWith({
      voice: {
        input_mode: 'legacy',
        realtime: {
          enabled: false,
          stt_provider: 'openai'
        },
        stop_phrases: ['enough']
      }
    })
  })
})
