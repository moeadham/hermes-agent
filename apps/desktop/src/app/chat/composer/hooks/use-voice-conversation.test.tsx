import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { BargeMonitorCallbacks } from '@/lib/voice-barge-in'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceConversation } from './use-voice-conversation'

// The full-duplex contract: the barge monitor is live across the WHOLE agent
// turn — generation (thinking) and playback (speaking) — so speaking over the
// model interrupts it mid-generation instead of the mic being deaf until TTS
// starts (the Windows report: interruption "never works" because the deaf
// window covered generation, and playback bleed made the old monitor's
// trigger unreachable).

const monitorCalls: BargeMonitorCallbacks[] = []
const stopMonitor = vi.fn()

vi.mock('@/lib/voice-barge-in', () => ({
  monitorSpeechDuringPlayback: (callbacks: BargeMonitorCallbacks) => {
    monitorCalls.push(callbacks)

    return stopMonitor
  }
}))

const markVoicePlaybackInterrupted = vi.fn()
const stopVoicePlayback = vi.fn()

vi.mock('@/lib/voice-playback', () => ({
  markVoicePlaybackInterrupted: () => markVoicePlaybackInterrupted(),
  playSpeechText: vi.fn(async () => true),
  startSpeechStream: vi.fn(async () => null),
  stopVoicePlayback: () => stopVoicePlayback()
}))

const realtimeInstances: Array<{
  callbacks: {
    onPartialTranscript?: (transcript: { id: string; text: string }) => void
    onSpeechStarted?: () => void
    onTranscript: (transcript: { id: string; text: string }) => void
  }
  connect: ReturnType<typeof vi.fn>
  disconnect: ReturnType<typeof vi.fn>
  setMuted: ReturnType<typeof vi.fn>
}> = []

vi.mock('@/lib/realtime-voice-session', () => ({
  RealtimeVoiceSession: class {
    callbacks: (typeof realtimeInstances)[number]['callbacks']
    connect = vi.fn(async () => undefined)
    disconnect = vi.fn()
    setMuted = vi.fn()

    constructor(callbacks: (typeof realtimeInstances)[number]['callbacks']) {
      this.callbacks = callbacks
      realtimeInstances.push(this)
    }
  }
}))

vi.mock('@/lib/thinking-sound', () => ({
  startThinkingSound: vi.fn(),
  stopThinkingSound: vi.fn()
}))

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => undefined),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => null)
}

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: micHandle, level: 0, recording: false })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'configure STT',
          couldNotStartSession: 'could not start',
          microphoneFailed: 'mic failed',
          playbackFailed: 'playback failed',
          transcriptionFailed: 'transcription failed',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

interface HookProps {
  busy: boolean
}

function renderConversation(overrides: { onInterrupt?: () => void; transcript?: string } = {}) {
  const onInterrupt = overrides.onInterrupt ?? vi.fn()

  // Mirrors the real app: submitting a turn makes the agent busy.
  const onBusyChange: { current: (busy: boolean) => void } = { current: () => undefined }

  const onSubmit = vi.fn(async () => {
    onBusyChange.current(true)
  })

  const onStopWord = vi.fn()

  // First transcription is the turn that starts the conversation; subsequent
  // ones are barge captures (the overridable transcript).
  let transcriptions = 0

  const onTranscribeAudio = vi.fn(async () =>
    transcriptions++ === 0 ? 'kick off the task' : (overrides.transcript ?? 'and another thing')
  )

  const hook = renderHook(
    ({ busy }: HookProps) =>
      useVoiceConversation({
        busy,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onInterrupt,
        onStopWord,
        onSubmit,
        onTranscribeAudio,
        pendingResponse: () => null
      }),
    { initialProps: { busy: false } }
  )

  onBusyChange.current = busy => hook.rerender({ busy })

  return { hook, onInterrupt, onStopWord, onSubmit, onTranscribeAudio }
}

/** Drive the hook into the generation phase (turn submitted, model working). */
async function enterThinking(hook: ReturnType<typeof renderConversation>['hook']) {
  await act(async () => {
    await hook.result.current.start()
  })
  await waitFor(() => expect(hook.result.current.status).toBe('listening'))

  micHandle.stop.mockResolvedValueOnce({
    audio: new Blob(['q'], { type: 'audio/webm' }),
    durationMs: 900,
    heardSpeech: true
  })

  await act(async () => {
    hook.result.current.stopTurn()
  })
  await waitFor(() => expect(hook.result.current.status).toBe('thinking'))
}

describe('useVoiceConversation full-duplex barge-in', () => {
  beforeEach(() => {
    monitorCalls.length = 0
    realtimeInstances.length = 0
    vi.clearAllMocks()
    micHandle.start.mockResolvedValue(undefined)
    micHandle.stop.mockResolvedValue(null)
  })

  afterEach(cleanup)

  it('arms the barge monitor during generation (before any reply audio exists)', async () => {
    const { hook } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)

    await waitFor(() => expect(hook.result.current.status).toBe('thinking'))
    // busy=true + thinking → the full-duplex monitor must be live.
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))
  })

  it('interrupts the in-flight turn when speech trips mid-generation', async () => {
    const { hook, onInterrupt } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    act(() => {
      monitorCalls.at(-1)?.onSpeech()
    })

    expect(onInterrupt).toHaveBeenCalledTimes(1)
    expect(markVoicePlaybackInterrupted).toHaveBeenCalled()
    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('submits the captured interruption once the interrupt settles (busy clears)', async () => {
    const { hook, onSubmit } = renderConversation({ transcript: 'no, do it differently' })

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const monitor = monitorCalls.at(-1)

    act(() => {
      monitor?.onSpeech()
    })

    // Interrupt lands → the turn ends → busy flips false.
    hook.rerender({ busy: false })

    await act(async () => {
      monitor?.onUtterance?.(new Blob(['x'], { type: 'audio/webm' }))
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('no, do it differently'))
  })

  it('does not interrupt when speech trips during playback (turn already done)', async () => {
    const { hook, onInterrupt } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    // Turn finished; playback phase.
    hook.rerender({ busy: false })

    act(() => {
      monitorCalls.at(-1)?.onSpeech()
    })

    expect(onInterrupt).not.toHaveBeenCalled()
    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('a spoken stop command in the barge capture ends the conversation instead of submitting', async () => {
    const { hook, onStopWord, onSubmit } = renderConversation({ transcript: 'stop' })

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const monitor = monitorCalls.at(-1)

    act(() => {
      monitor?.onSpeech()
    })
    hook.rerender({ busy: false })

    await act(async () => {
      monitor?.onUtterance?.(new Blob(['s'], { type: 'audio/webm' }))
    })

    await waitFor(() => expect(onStopWord).toHaveBeenCalledTimes(1))
    // Only the kickoff turn was submitted — the "stop" capture never was.
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalledWith('stop')
  })

  it('re-arms a single monitor per turn (idempotent ensure)', async () => {
    const { hook } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const armed = monitorCalls.length

    // Effect re-runs (busy toggles, status changes) must not open more mics.
    hook.rerender({ busy: true })
    hook.rerender({ busy: true })

    expect(monitorCalls.length).toBe(armed)
  })

  it('uses realtime transport in realtime mode and submits only finalized transcripts', async () => {
    const onSubmit = vi.fn()

    const hook = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        mode: 'realtime',
        onSubmit,
        onTranscribeAudio: vi.fn(async () => 'legacy should not run'),
        pendingResponse: () => null,
        realtimeProvider: 'elevenlabs',
        sessionId: 'desktop-session'
      })
    )

    await act(async () => {
      await hook.result.current.start()
    })

    expect(realtimeInstances).toHaveLength(1)
    expect(realtimeInstances[0]!.connect).toHaveBeenCalledWith({
      provider: 'elevenlabs',
      sessionId: 'desktop-session'
    })

    act(() => {
      realtimeInstances[0]!.callbacks.onPartialTranscript?.({ id: 'seg-1', text: 'hel' })
      realtimeInstances[0]!.callbacks.onTranscript({ id: 'seg-1', text: 'hello' })
      realtimeInstances[0]!.callbacks.onTranscript({ id: 'seg-1', text: 'hello' })
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit).toHaveBeenCalledWith('hello')
  })

  it('ends realtime voice on a finalized stop command instead of submitting it', async () => {
    const onSubmit = vi.fn()
    const onStopWord = vi.fn()

    const hook = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        mode: 'realtime',
        onStopWord,
        onSubmit,
        pendingResponse: () => null
      })
    )

    await act(async () => {
      await hook.result.current.start()
    })

    act(() => {
      realtimeInstances[0]!.callbacks.onTranscript({ id: 'seg-stop', text: 'never mind' })
    })

    await waitFor(() => expect(onStopWord).toHaveBeenCalledTimes(1))

    expect(onSubmit).not.toHaveBeenCalled()

    expect(realtimeInstances[0]!.disconnect).toHaveBeenCalledTimes(1)

    expect(hook.result.current.status).toBe('idle')
  })

  it('does not arm the legacy barge monitor while realtime provider mic owns barge-in', async () => {
    const onInterrupt = vi.fn()

    const pendingResponseState: { current: { id: string; pending: boolean; text: string } | null } = {
      current: null
    }

    const onBusyChange: { current: (busy: boolean) => void } = { current: () => undefined }

    const onSubmit = vi.fn(() => {
      pendingResponseState.current = { id: 'reply-1', pending: true, text: '' }
      onBusyChange.current(true)
    })

    const hook = renderHook(
      ({ busy }: HookProps) =>
        useVoiceConversation({
          busy,
          consumePendingResponse: vi.fn(),
          enabled: true,
          mode: 'realtime',
          onInterrupt,
          onSubmit,
          pendingResponse: () => pendingResponseState.current
        }),
      { initialProps: { busy: false } }
    )

    onBusyChange.current = busy => hook.rerender({ busy })

    await act(async () => {
      await hook.result.current.start()
    })

    act(() => {
      realtimeInstances[0]!.callbacks.onTranscript({ id: 'seg-1', text: 'change course' })
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('change course'))
    await waitFor(() => expect(hook.result.current.status).toBe('speaking'))
    expect(monitorCalls).toHaveLength(0)

    act(() => {
      realtimeInstances[0]!.callbacks.onSpeechStarted?.()
    })

    expect(onInterrupt).toHaveBeenCalledTimes(1)
    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('submits a realtime final that arrives immediately after speech-start while interrupt is settling', async () => {
    let settleInterrupt!: () => void

    const onInterrupt = vi.fn(
      () =>
        new Promise<void>(resolve => {
          settleInterrupt = resolve
        })
    )

    let currentBusy = false

    const acceptedSubmissions: string[] = []
    const droppedSubmissions: string[] = []

    const onSubmit = vi.fn((text: string) => {
      if (currentBusy) {
        droppedSubmissions.push(text)

        return
      }

      acceptedSubmissions.push(text)
    })

    const hook = renderHook(
      ({ busy }: HookProps) =>
        useVoiceConversation({
          busy,
          consumePendingResponse: vi.fn(),
          enabled: true,
          mode: 'realtime',
          onInterrupt,
          onSubmit,
          pendingResponse: () => null
        }),
      { initialProps: { busy: false } }
    )

    await act(async () => {
      await hook.result.current.start()
    })

    await act(async () => {
      currentBusy = true
      hook.rerender({ busy: true })
    })

    act(() => {
      realtimeInstances[0]!.callbacks.onSpeechStarted?.()
      realtimeInstances[0]!.callbacks.onTranscript({ id: 'seg-barge', text: 'actually use this' })
      realtimeInstances[0]!.callbacks.onTranscript({ id: 'seg-barge', text: 'actually use this' })
    })

    expect(onInterrupt).toHaveBeenCalledTimes(1)
    expect(acceptedSubmissions).toEqual([])

    await act(async () => {
      settleInterrupt()
    })

    await act(async () => {
      currentBusy = false
      hook.rerender({ busy: false })
    })

    await waitFor(() => expect(acceptedSubmissions).toEqual(['actually use this']))
    expect(droppedSubmissions).toEqual([])
  })
})
