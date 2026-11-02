# 任意の実験用セットアップと画面の復旧

通常のdoctor・推論の前提にはしない。日常用の設定と実験用の設定を自分で選び、戻せる形で扱う。

## 給電中の画面維持

    ./pixel-lab session status
    ./pixel-lab session prepare
    ./pixel-lab session check
    ./pixel-lab session restore

`prepare` は元の2設定をMacのペアリング用状態ディレクトリに保存してから、AC/USB給電中の画面維持と30分の画面タイムアウトを要求する。バックアップが残っている間は再上書きしない。`restore` は同じUSB端末であることを照合して元に戻す。

設定要求と実効値は異なる。2026-11-03の検証端末には仕事用プロファイルの管理元が個人側に指定した60秒上限があり、要求後も `admin_max_screen_off_ms=60000`、`stay_on_effective=false` だった。端末側の設定だけではこの上限を緩和できなかった。PINの変更・管理アプリの削除・仕事用データの消去は行わない。

管理元を変更できる場合は、管理コンソールの画面ロック／非操作時間の設定を変更して端末への反映を確認する。端末だけで運用する場合、仕事用プロファイルの一時停止は背景処理を減らせるが、個人側の上限まで消えるとは限らない。独立したAndroid実験用ユーザーも候補だが、Linux VMと鍵はユーザーごとになるので別途セットアップが必要。これらを自動の前提条件にはしない。

[Googleの画面維持API](https://developer.android.com/develop/background-work/background-tasks/awake/screen-on) · [管理ポリシーの最大ロック時間](https://developer.android.com/reference/android/app/admin/DevicePolicyManager#setMaximumTimeToLock(android.content.ComponentName,long))

## Googleの背景同期を減らす

端末の「設定」→「パスワード、パスキー、自動入力」または「アカウント」から、個人データの自動同期、またはアカウントごとの同期項目をオフにする。表示名はAndroidの版によって異なる。仕事用アプリをまとめて止めたい場合は、アプリ一覧の「仕事」タブ、またはクイック設定の仕事用プロファイルを一時停止する。戻すときは同じスイッチをオンにする。

自動同期を止めると、対象アプリの更新や通知が手動同期まで遅れる。Google Play開発者サービス全体、push配信、各アプリのバックアップ等の全通信を止める設定ではない。検証端末では個人側のmaster auto-syncがtrueからfalseになったことを確認した。仕事側は別の設定であり、変更済みとは扱わない。

[Googleアカウントの同期設定](https://support.google.com/pixelphone/answer/2840875?hl=ja) · [仕事用プロファイルの一時停止](https://support.google.com/pixelphone/answer/7029561?hl=ja)

確認に `dumpsys content` を使う場合、生出力にはメールアドレスなどが含まれる。Repoに保存せず、必要な真偽値だけを取り出す。

## 小さく実行して停止を確認する

    ./pixel-lab session run --seconds 20 -- /absolute/guest/program arg1 arg2

実行前・実行中・実行後に、ADBとPNG画面キャプチャの応答を確認する。画面そのものは保存せず、寸法・ハッシュ・電源状態だけを記録する。通常のロック状態でも実行できる。画面操作が必要な検査だけ `--require-unlocked` を付ける。

コマンドはVM側の独立したプロセスグループで動く。VM自身が期限を持ち、超過またはcancelファイルでTERM→KILLを送る。SSH切断のHUPでは監督プロセスを終了させず、期限を継続する。上限は120秒、既定値は20秒。画面応答が途切れたらホストはcancelを要求し、届かなかった場合もVM側の期限が残る。

Macの `runs/session-*/` に `session.json`、`job.json`、標準出力・標準エラー・接続ログを残す。失敗・期限超過・回収不完了は成功にしない。これはGPUドライバのハングを防ぎ切る保証ではない。カーネル内部で停止している処理はユーザー空間のKILLでも即座に終了しない場合がある。最初は小さなバッファの転送検査で使い、コンパイルとGPU負荷を同時に走らせない。

## 黒画面になったとき

    ./pixel-lab session recover
    ./pixel-lab session recover --stop-vm
    ./pixel-lab session recover --reboot

最初は画面を起こしてホームへ戻り、キャプチャ応答を確認する。`--stop-vm` は標準TerminalとVM内の実行中の処理を停止するので、未保存の作業は失われる。描画が戻らないときの `--reboot` は通常のAndroid再起動。再起動後の本人による解除は必要。監視処理は勝手に再起動しない。

**実機検証：** 状態取得・設定保存・要求・復元、PNG応答、短いVMコマンドの正常終了と期限停止を確認。意図的なGPUハングは起こしていない。復旧用rebootは、先行する黒画面から通常再起動して実画面の表示と解除が戻った手順を明示的なコマンドにしたもの。
