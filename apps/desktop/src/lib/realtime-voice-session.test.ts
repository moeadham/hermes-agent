// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { RealtimeVoiceSession, reduceRealtimeVoiceEvent } from './realtime-voice-session'

class FakeTrack {
  enabled = true
  stop = vi.fn()
}

class FakeStream {
  readonly track = new FakeTrack()

  getAudioTracks() {
    return [this.track] as unknown as MediaStreamTrack[]
  }

  getTracks() {
    return this.getAudioTracks()
  }
}

class FakeDataChannel extends EventTarget {
  readyState: RTCDataChannelState = 'open'
  readonly sent: string[] = []
  close = vi.fn(() => {
    this.readyState = 'closed'
  })

  send(value: string) {
    this.sent.push(value)
  }

  emit(event: unknown) {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(event) }))
  }
}

class FakePeer extends EventTarget {
  readonly channel = new FakeDataChannel()
  connectionState: RTCPeerConnectionState = 'new'
  close = vi.fn(() => {
    this.connectionState = 'closed'
  })
  addTrack = vi.fn()
  createDataChannel = vi.fn(() => this.channel as unknown as RTCDataChannel)
  createOffer = vi.fn(async () => ({ sdp: 'offer-sdp', type: 'offer' }) as RTCSessionDescriptionInit)
  setLocalDescription = vi.fn(async () => undefined)
  setRemoteDescription = vi.fn(async () => undefined)
}

class FakeWebSocket extends EventTarget {
  static instances: FakeWebSocket[] = []
  readyState: number = WebSocket.CONNECTING
  readonly sent: Array<string | ArrayBufferLike | Blob | ArrayBufferView> = []
  close = vi.fn(() => {
    this.readyState = WebSocket.CLOSED
  })

  constructor(readonly url: string) {
    super()
    FakeWebSocket.instances.push(this)
  }

  send(value: string | ArrayBufferLike | Blob | ArrayBufferView) {
    this.sent.push(value)
  }

  open() {
    this.readyState = WebSocket.OPEN
    this.dispatchEvent(new Event('open'))
  }

  emit(event: unknown) {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(event) }))
  }
}

class FakeMediaRecorder extends EventTarget {
  static instances: FakeMediaRecorder[] = []
  state: RecordingState = 'inactive'

  constructor() {
    super()
    FakeMediaRecorder.instances.push(this)
  }

  start = vi.fn(() => {
    this.state = 'recording'
  })
  stop = vi.fn(() => {
    this.state = 'inactive'
  })
}

class FakeAudioNode {
  connect = vi.fn()
  disconnect = vi.fn()
}

class FakeScriptProcessor extends FakeAudioNode {
  onaudioprocess: ((event: AudioProcessingEvent) => void) | null = null
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []
  destination = new FakeAudioNode()
  processor = new FakeScriptProcessor()
  sampleRate = 48000
  close = vi.fn(async () => undefined)
  createMediaStreamSource = vi.fn(() => new FakeAudioNode())
  createScriptProcessor = vi.fn(() => this.processor)
  resume = vi.fn(async () => undefined)

  constructor() {
    FakeAudioContext.instances.push(this)
  }
}

function fixture(provider: 'openai' | 'elevenlabs' = 'openai') {
  const peers: FakePeer[] = []
  const streams: FakeStream[] = []
  const partials = vi.fn()
  const transcripts = vi.fn()
  const onError = vi.fn()
  const onSpeechStarted = vi.fn()
  const fetchImpl = vi.fn(async () => new Response('answer-sdp', { status: 200 }))

  const createSession = vi.fn(async () =>
    provider === 'openai'
      ? {
          client_secret: 'ek_short_lived_renderer_secret',
          expires_at: Math.floor(Date.now() / 1000) + 60,
          ok: true,
          provider: 'openai',
          session_binding: 'binding-1'
        }
      : {
          bridge_url: 'ws://testserver/api/audio/realtime/elevenlabs/ws?token=bridge',
          expires_at: Math.floor(Date.now() / 1000) + 60,
          ok: true,
          provider: 'elevenlabs',
          session_binding: 'binding-2'
        }
  )

  const session = new RealtimeVoiceSession(
    { onError, onPartialTranscript: partials, onSpeechStarted, onTranscript: transcripts },
    {
      createPeerConnection: () => {
        const peer = new FakePeer()
        peers.push(peer)

        return peer as unknown as RTCPeerConnection
      },
      createSession,
      fetch: fetchImpl as unknown as typeof fetch,
      getUserMedia: async () => {
        const stream = new FakeStream()
        streams.push(stream)

        return stream as unknown as MediaStream
      },
      MediaRecorderCtor: FakeMediaRecorder as unknown as typeof MediaRecorder,
      WebSocketCtor: FakeWebSocket as unknown as typeof WebSocket
    }
  )

  return { createSession, fetchImpl, onError, onSpeechStarted, partials, peers, session, streams, transcripts }
}

describe('RealtimeVoiceSession', () => {
  beforeEach(() => {
    FakeWebSocket.instances.length = 0
    FakeMediaRecorder.instances.length = 0
    FakeAudioContext.instances.length = 0
    vi.restoreAllMocks()
    vi.stubGlobal('AudioContext', FakeAudioContext)
  })

  it('uses OpenAI ephemeral WebRTC credentials and submits finals only once', async () => {
    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    const { createSession, fetchImpl, peers, partials, session, transcripts } = fixture('openai')

    await session.connect({ language: 'ko', provider: 'openai', sessionId: 'desktop-session-1' })

    expect(createSession).toHaveBeenCalledWith('desktop-session-1', 'ko', 'openai')
    expect(fetchImpl).toHaveBeenCalledWith(
      'https://api.openai.com/v1/realtime/calls',
      expect.objectContaining({
        body: 'offer-sdp',
        headers: {
          Authorization: 'Bearer ek_short_lived_renderer_secret',
          'Content-Type': 'application/sdp'
        },
        method: 'POST'
      })
    )
    expect(setItem).not.toHaveBeenCalled()

    peers[0]!.channel.emit({
      type: 'conversation.item.input_audio_transcription.delta',
      item_id: 'item-1',
      delta: 'hel'
    })
    peers[0]!.channel.emit({
      type: 'conversation.item.input_audio_transcription.completed',
      item_id: 'item-1',
      transcript: 'hello'
    })
    peers[0]!.channel.emit({
      type: 'conversation.item.input_audio_transcription.completed',
      item_id: 'item-1',
      transcript: 'hello'
    })

    expect(partials).toHaveBeenCalledWith({ id: 'item-1', text: 'hel' })
    expect(transcripts).toHaveBeenCalledTimes(1)
    expect(transcripts).toHaveBeenCalledWith({ id: 'item-1', text: 'hello' })
  })

  it('stops the acquired microphone stream when OpenAI setup throws after getUserMedia succeeds', async () => {
    const stream = new FakeStream()
    const onError = vi.fn()

    const session = new RealtimeVoiceSession(
      { onError, onTranscript: vi.fn() },
      {
        createPeerConnection: () => {
          throw new Error('peer setup failed')
        },
        createSession: vi.fn(async () => ({
          client_secret: 'ek_short_lived_renderer_secret',
          expires_at: Math.floor(Date.now() / 1000) + 60,
          ok: true,
          provider: 'openai',
          session_binding: 'binding-1'
        })),
        getUserMedia: vi.fn(async () => stream as unknown as MediaStream),
        MediaRecorderCtor: FakeMediaRecorder as unknown as typeof MediaRecorder,
        WebSocketCtor: FakeWebSocket as unknown as typeof WebSocket
      }
    )

    await session.connect({ provider: 'openai', sessionId: 'desktop-session-setup-fail' })

    expect(stream.track.stop).toHaveBeenCalledTimes(1)
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: 'peer setup failed' }))
  })

  it('uses the ElevenLabs backend bridge instead of exposing a provider key', async () => {
    const { createSession, session, transcripts } = fixture('elevenlabs')

    await session.connect({ provider: 'elevenlabs', sessionId: 'desktop-session-2' })
    FakeWebSocket.instances[0]!.open()

    expect(createSession).toHaveBeenCalledWith('desktop-session-2', undefined, 'elevenlabs')
    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(FakeWebSocket.instances[0]!.url).toBe('ws://testserver/api/audio/realtime/elevenlabs/ws?token=bridge')
    expect(FakeWebSocket.instances[0]!.url).not.toContain('ELEVENLABS')

    FakeWebSocket.instances[0]!.emit({ type: 'partial_transcript', id: 'seg-1', transcript: 'hel' })
    FakeWebSocket.instances[0]!.emit({ type: 'committed_transcript', segment_id: 'seg-1', text: 'hello' })

    expect(transcripts).toHaveBeenCalledWith({ id: 'seg-1', text: 'hello' })
  })

  it('sends genuine 16 kHz mono little-endian PCM frames to the ElevenLabs bridge', async () => {
    const { session } = fixture('elevenlabs')

    await session.connect({ provider: 'elevenlabs', sessionId: 'desktop-session-2' })
    FakeWebSocket.instances[0]!.open()

    expect(FakeMediaRecorder.instances).toHaveLength(0)
    expect(FakeAudioContext.instances).toHaveLength(1)

    const audioContext = FakeAudioContext.instances[0]!
    audioContext.processor.onaudioprocess?.({
      inputBuffer: {
        getChannelData: () => Float32Array.from([0, 0.5, -0.5, 1, -1, 0]),
        length: 6,
        numberOfChannels: 1
      }
    } as unknown as AudioProcessingEvent)

    expect(FakeWebSocket.instances[0]!.sent).toHaveLength(1)
    expect(FakeWebSocket.instances[0]!.sent[0]).toBeInstanceOf(ArrayBuffer)

    const samples = Array.from(new Int16Array(FakeWebSocket.instances[0]!.sent[0] as unknown as ArrayBuffer))
    expect(samples).toEqual([0, 32767])
  })

  it.each(['vad_speech_start', 'speech_start', 'speech_started', 'barge_in', 'interruption'])(
    'treats ElevenLabs %s bridge events as speech start',
    async type => {
      const { onError, onSpeechStarted, partials, session, transcripts } = fixture('elevenlabs')

      await session.connect({ provider: 'elevenlabs', sessionId: 'desktop-session-barge' })
      FakeWebSocket.instances[0]!.open()

      FakeWebSocket.instances[0]!.emit({ type, text: 'ignored until final transcript' })

      expect(onSpeechStarted).toHaveBeenCalledTimes(1)
      expect(partials).not.toHaveBeenCalled()
      expect(transcripts).not.toHaveBeenCalled()
      expect(onError).not.toHaveBeenCalled()
      expect(session.currentStatus).toBe('listening')
    }
  )
})

describe('reduceRealtimeVoiceEvent', () => {
  it('normalizes provider events and rejects response generation', () => {
    expect(
      reduceRealtimeVoiceEvent({
        type: 'conversation.item.input_audio_transcription.completed',
        item_id: 'item-1',
        transcript: '  hello  '
      })
    ).toEqual({ status: 'listening', transcript: { id: 'item-1', text: 'hello' } })
    expect(reduceRealtimeVoiceEvent({ type: 'partial_transcript', id: 'seg-1', transcript: 'he' })).toEqual({
      partialTranscript: { id: 'seg-1', text: 'he' },
      status: 'listening'
    })
    expect(reduceRealtimeVoiceEvent({ type: 'response.audio.delta' })).toEqual({
      error: 'Unexpected Realtime response-generation event',
      status: 'error'
    })
  })

  it.each(['vad_speech_start', 'speech_start', 'speech_started', 'barge_in', 'interruption'])(
    'normalizes ElevenLabs %s as one speech-start action',
    type => {
      expect(reduceRealtimeVoiceEvent({ type, text: 'ignored until final transcript' })).toEqual({
        speechStarted: true,
        status: 'listening'
      })
    }
  )

  it('keeps ElevenLabs partial, final, end, and error variants distinct from speech start', () => {
    expect(reduceRealtimeVoiceEvent({ type: 'partial_transcript', id: 'seg-1', transcript: 'he' })).toEqual({
      partialTranscript: { id: 'seg-1', text: 'he' },
      status: 'listening'
    })
    expect(reduceRealtimeVoiceEvent({ type: 'committed_transcript', segment_id: 'seg-1', text: 'hello' })).toEqual({
      status: 'listening',
      transcript: { id: 'seg-1', text: 'hello' }
    })
    expect(reduceRealtimeVoiceEvent({ type: 'speech_end' })).toEqual({ status: 'transcribing' })
    expect(reduceRealtimeVoiceEvent({ type: 'transcriber_error' })).toEqual({
      error: 'Realtime transcription failed',
      status: 'error'
    })
  })
})
