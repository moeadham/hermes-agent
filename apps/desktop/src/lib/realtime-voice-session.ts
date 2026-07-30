import { createRealtimeVoiceSession } from '@/hermes'

const OPENAI_REALTIME_CALLS_URL = 'https://api.openai.com/v1/realtime/calls'

export type RealtimeVoiceProvider = 'openai' | 'elevenlabs'
export type RealtimeVoiceStatus = 'idle' | 'connecting' | 'listening' | 'transcribing' | 'error'

export interface RealtimeTranscript {
  id: string
  text: string
}

export interface RealtimeVoiceEventAction {
  error?: string
  partialTranscript?: RealtimeTranscript
  speechStarted?: boolean
  status?: RealtimeVoiceStatus
  transcript?: RealtimeTranscript
}

function textFrom(record: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) {
    const value = record[key]

    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return ''
}

function idFrom(record: Record<string, unknown>, ...keys: string[]): string {
  return textFrom(record, ...keys)
}

export function reduceRealtimeVoiceEvent(event: unknown): RealtimeVoiceEventAction {
  if (!event || typeof event !== 'object') {
    return { error: 'Invalid Realtime voice event', status: 'error' }
  }

  const record = event as Record<string, unknown>

  const type =
    typeof record.type === 'string'
      ? record.type
      : typeof record.message_type === 'string'
        ? record.message_type
        : ''

  const normalized = type.toLowerCase().replaceAll('.', '_')

  if (type.startsWith('response.') || type.includes('function_call')) {
    return { error: 'Unexpected Realtime response-generation event', status: 'error' }
  }

  if (
    type === 'input_audio_buffer.speech_started' ||
    ['vad_speech_start', 'speech_start', 'speech_started', 'barge_in', 'interruption'].includes(normalized)
  ) {
    return { speechStarted: true, status: 'listening' }
  }

  if (
    type === 'input_audio_buffer.speech_stopped' ||
    ['vad_speech_end', 'speech_end', 'speech_stopped', 'commit', 'committed'].includes(normalized)
  ) {
    return { status: 'transcribing' }
  }

  if (
    type === 'conversation.item.input_audio_transcription.delta' ||
    type === 'conversation.item.input_audio_transcription.partial' ||
    ['partial_transcript', 'transcript_partial', 'partial'].includes(normalized)
  ) {
    const id = idFrom(record, 'item_id', 'segment_id', 'id')
    const text = textFrom(record, 'delta', 'transcript', 'text')

    return id && text ? { partialTranscript: { id, text }, status: 'listening' } : {}
  }

  if (
    type === 'conversation.item.input_audio_transcription.completed' ||
    [
      'final_transcript',
      'committed_transcript',
      'committed_transcript_with_timestamps',
      'transcript_final',
      'transcript_committed',
      'final'
    ].includes(normalized)
  ) {
    const id = idFrom(record, 'item_id', 'segment_id', 'id')
    const text = textFrom(record, 'transcript', 'text')

    return id && text ? { status: 'listening', transcript: { id, text } } : {}
  }

  if (
    type === 'conversation.item.input_audio_transcription.failed' ||
    [
      'auth_error',
      'chunk_size_exceeded',
      'commit_throttled',
      'error',
      'input_error',
      'quota_exceeded',
      'rate_limited',
      'resource_exhausted',
      'session_time_limit_exceeded',
      'transcriber_error',
      'unaccepted_terms'
    ].includes(normalized)
  ) {
    return { error: 'Realtime transcription failed', status: 'error' }
  }

  return {}
}

interface RealtimeVoiceCallbacks {
  onError?: (error: Error) => void
  onPartialTranscript?: (transcript: RealtimeTranscript) => void
  onSpeechStarted?: () => void
  onStatus?: (status: RealtimeVoiceStatus) => void
  onTranscript: (transcript: RealtimeTranscript) => void
}

interface RealtimeVoiceDependencies {
  AudioContextCtor?: typeof AudioContext
  createPeerConnection?: () => RTCPeerConnection
  createSession?: typeof createRealtimeVoiceSession
  fetch?: typeof fetch
  getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>
  MediaRecorderCtor?: typeof MediaRecorder
  WebSocketCtor?: typeof WebSocket
}

export interface RealtimeVoiceConnectOptions {
  language?: string
  provider?: RealtimeVoiceProvider
  sessionId: string
}

export class RealtimeVoiceSession {
  private channel: RTCDataChannel | null = null
  private generation = 0
  private readonly seenTranscriptIds = new Set<string>()
  private audioContext: AudioContext | null = null
  private audioProcessor: ScriptProcessorNode | null = null
  private audioSource: MediaStreamAudioSourceNode | null = null
  private peer: RTCPeerConnection | null = null
  private status: RealtimeVoiceStatus = 'idle'
  private recorder: MediaRecorder | null = null
  private stream: MediaStream | null = null
  private ws: WebSocket | null = null

  private readonly AudioContextCtor: typeof AudioContext
  private readonly createPeerConnection: () => RTCPeerConnection
  private readonly createSession: typeof createRealtimeVoiceSession
  private readonly fetchImpl: typeof fetch
  private readonly getUserMedia: (constraints: MediaStreamConstraints) => Promise<MediaStream>
  private readonly MediaRecorderCtor: typeof MediaRecorder
  private readonly WebSocketCtor: typeof WebSocket

  constructor(
    private readonly callbacks: RealtimeVoiceCallbacks,
    dependencies: RealtimeVoiceDependencies = {}
  ) {
    this.AudioContextCtor =
      dependencies.AudioContextCtor ??
      globalThis.AudioContext ??
      (globalThis as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext!
    this.createPeerConnection = dependencies.createPeerConnection ?? (() => new RTCPeerConnection())
    this.createSession = dependencies.createSession ?? createRealtimeVoiceSession
    this.fetchImpl = dependencies.fetch ?? fetch
    this.getUserMedia = dependencies.getUserMedia ?? (constraints => navigator.mediaDevices.getUserMedia(constraints))
    this.MediaRecorderCtor = dependencies.MediaRecorderCtor ?? MediaRecorder
    this.WebSocketCtor = dependencies.WebSocketCtor ?? WebSocket
  }

  get currentStatus() {
    return this.status
  }

  async connect({ language, provider = 'openai', sessionId }: RealtimeVoiceConnectOptions): Promise<void> {
    this.disconnect()
    const ownGeneration = this.generation
    this.setStatus('connecting')

    try {
      const session = await this.createSession(sessionId, language, provider)

      if (ownGeneration !== this.generation) {
        return
      }

      if (provider === 'elevenlabs') {
        await this.connectElevenLabsBridge(session, ownGeneration)

        return
      }

      await this.connectOpenAIWebRtc(session, ownGeneration)
    } catch (error) {
      if (ownGeneration !== this.generation) {
        return
      }

      this.fail(error)
    }
  }

  cancelInput() {
    if (this.channel?.readyState === 'open') {
      this.channel.send(JSON.stringify({ type: 'input_audio_buffer.clear' }))
    }

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'clear' }))
    }
  }

  setMuted(muted: boolean) {
    this.stream?.getAudioTracks().forEach(track => {
      track.enabled = !muted
    })
  }

  disconnect() {
    this.generation += 1
    this.closeResources()
    this.seenTranscriptIds.clear()
    this.setStatus('idle')
  }

  private async connectOpenAIWebRtc(session: Awaited<ReturnType<typeof createRealtimeVoiceSession>>, generation: number) {
    if (!session.client_secret || session.expires_at * 1000 <= Date.now() + 5_000 || !session.session_binding) {
      throw new Error('Realtime voice session secret is invalid or expired')
    }

    const stream = await this.getUserMedia({
      audio: {
        autoGainControl: true,
        echoCancellation: true,
        noiseSuppression: true
      }
    })

    if (generation !== this.generation) {
      stream.getTracks().forEach(track => track.stop())

      return
    }

    this.stream = stream
    const peer = this.createPeerConnection()
    const channel = peer.createDataChannel('oai-events')

    this.peer = peer
    this.channel = channel
    stream.getAudioTracks().forEach(track => peer.addTrack(track, stream))
    channel.addEventListener('message', this.handleMessage)
    channel.addEventListener('close', this.handleChannelClose)
    peer.addEventListener('connectionstatechange', this.handleConnectionState)

    const offer = await peer.createOffer()
    await peer.setLocalDescription(offer)

    const response = await this.fetchImpl(OPENAI_REALTIME_CALLS_URL, {
      method: 'POST',
      body: offer.sdp ?? '',
      headers: {
        Authorization: `Bearer ${session.client_secret}`,
        'Content-Type': 'application/sdp'
      }
    })

    if (!response.ok) {
      throw new Error(`Realtime WebRTC negotiation failed (${response.status})`)
    }

    const answerSdp = await response.text()

    if (!answerSdp) {
      throw new Error('Realtime WebRTC negotiation returned an empty answer')
    }

    if (generation !== this.generation) {
      return
    }

    await peer.setRemoteDescription({ type: 'answer', sdp: answerSdp })
  }

  private async connectElevenLabsBridge(session: Awaited<ReturnType<typeof createRealtimeVoiceSession>>, generation: number) {
    if (!session.bridge_url || session.expires_at * 1000 <= Date.now() + 5_000 || !session.session_binding) {
      throw new Error('Realtime voice bridge is invalid or expired')
    }

    const stream = await this.getUserMedia({
      audio: {
        autoGainControl: true,
        echoCancellation: true,
        noiseSuppression: true
      }
    })

    if (generation !== this.generation) {
      stream.getTracks().forEach(track => track.stop())

      return
    }

    this.stream = stream
    const ws = new this.WebSocketCtor(session.bridge_url)

    this.ws = ws
    ws.addEventListener('message', this.handleMessage)
    ws.addEventListener('close', this.handleBridgeClose)
    ws.addEventListener('error', this.handleBridgeError)
    ws.addEventListener('open', () => {
      if (generation !== this.generation || this.ws !== ws) {
        return
      }

      void this.startElevenLabsPcmCapture(stream, ws, generation).catch(error => {
        if (generation === this.generation && this.ws === ws) {
          this.fail(error)
        }
      })
    })

    if (generation === this.generation) {
      this.setStatus('listening')
    }
  }

  private readonly handleMessage = (message: MessageEvent) => {
    let event: unknown

    try {
      event = JSON.parse(String(message.data))
    } catch {
      this.fail(new Error('Invalid Realtime voice event'))

      return
    }

    const action = reduceRealtimeVoiceEvent(event)

    if (action.error) {
      this.fail(new Error(action.error))

      return
    }

    if (action.status) {
      this.setStatus(action.status)
    }

    if (action.speechStarted) {
      this.callbacks.onSpeechStarted?.()
    }

    if (action.partialTranscript) {
      this.callbacks.onPartialTranscript?.(action.partialTranscript)
    }

    if (action.transcript && !this.seenTranscriptIds.has(action.transcript.id)) {
      this.seenTranscriptIds.add(action.transcript.id)
      this.callbacks.onTranscript(action.transcript)
    }
  }

  private async startElevenLabsPcmCapture(stream: MediaStream, ws: WebSocket, generation: number) {
    if (!this.AudioContextCtor) {
      throw new Error('Realtime voice PCM capture is unavailable in this browser')
    }

    const audioContext = new this.AudioContextCtor()
    const source = audioContext.createMediaStreamSource(stream)
    const processor = audioContext.createScriptProcessor(4096, 1, 1)

    this.audioContext = audioContext
    this.audioSource = source
    this.audioProcessor = processor

    processor.onaudioprocess = event => {
      if (generation !== this.generation || this.ws !== ws || ws.readyState !== WebSocket.OPEN) {
        return
      }

      const frame = pcm16Mono16000FromAudioBuffer(event.inputBuffer, audioContext.sampleRate)

      if (frame.byteLength > 0) {
        ws.send(frame)
      }
    }

    source.connect(processor)
    processor.connect(audioContext.destination)

    if (audioContext.state === 'suspended') {
      await audioContext.resume()
    }
  }

  private readonly handleChannelClose = () => {
    if (this.status !== 'idle') {
      this.fail(new Error('Realtime voice data channel closed'))
    }
  }

  private readonly handleBridgeClose = () => {
    if (this.status !== 'idle') {
      this.fail(new Error('Realtime voice bridge closed'))
    }
  }

  private readonly handleBridgeError = () => {
    this.fail(new Error('Realtime voice bridge failed'))
  }

  private readonly handleConnectionState = () => {
    const connectionState = this.peer?.connectionState

    if (connectionState === 'connected') {
      this.setStatus('listening')
    } else if (connectionState === 'failed' || connectionState === 'disconnected') {
      this.fail(new Error('Realtime voice connection was lost'))
    }
  }

  private fail(error: unknown) {
    const normalized = error instanceof Error ? error : new Error('Realtime voice session failed')

    this.closeResources()
    this.setStatus('error')
    this.callbacks.onError?.(normalized)
  }

  private closeResources() {
    if (this.channel) {
      this.channel.removeEventListener('message', this.handleMessage)
      this.channel.removeEventListener('close', this.handleChannelClose)
      this.channel.close()
      this.channel = null
    }

    if (this.peer) {
      this.peer.removeEventListener('connectionstatechange', this.handleConnectionState)
      this.peer.close()
      this.peer = null
    }

    if (this.ws) {
      this.ws.removeEventListener('message', this.handleMessage)
      this.ws.removeEventListener('close', this.handleBridgeClose)
      this.ws.removeEventListener('error', this.handleBridgeError)
      this.ws.close()
      this.ws = null
    }

    if (this.recorder) {
      if (this.recorder.state !== 'inactive') {
        this.recorder.stop()
      }

      this.recorder = null
    }

    if (this.audioProcessor) {
      this.audioProcessor.onaudioprocess = null
      this.audioProcessor.disconnect()
      this.audioProcessor = null
    }

    if (this.audioSource) {
      this.audioSource.disconnect()
      this.audioSource = null
    }

    if (this.audioContext) {
      void this.audioContext.close()
      this.audioContext = null
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop())
      this.stream = null
    }
  }

  private setStatus(status: RealtimeVoiceStatus) {
    if (this.status === status) {
      return
    }

    this.status = status
    this.callbacks.onStatus?.(status)
  }
}

function pcm16Mono16000FromAudioBuffer(inputBuffer: AudioBuffer, sourceSampleRate: number): ArrayBuffer {
  const channelCount = Math.max(1, inputBuffer.numberOfChannels)
  const mono = new Float32Array(inputBuffer.length)

  for (let channel = 0; channel < channelCount; channel += 1) {
    const input = inputBuffer.getChannelData(channel)

    for (let index = 0; index < mono.length; index += 1) {
      mono[index] += input[index] ?? 0
    }
  }

  for (let index = 0; index < mono.length; index += 1) {
    mono[index] /= channelCount
  }

  const targetSampleRate = 16000
  const outputLength = Math.floor((mono.length * targetSampleRate) / sourceSampleRate)
  const output = new Int16Array(outputLength)
  const ratio = sourceSampleRate / targetSampleRate

  for (let index = 0; index < outputLength; index += 1) {
    const position = index * ratio
    const left = Math.floor(position)
    const right = Math.min(left + 1, mono.length - 1)
    const weight = position - left
    const sample = (mono[left] ?? 0) * (1 - weight) + (mono[right] ?? 0) * weight
    const clamped = Math.max(-1, Math.min(1, sample))

    output[index] = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767)
  }

  return output.buffer
}
