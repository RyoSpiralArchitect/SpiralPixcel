# 128 MiB超の出力行列を調べる

**状態：完了待ちを修正した経路は全GPU要求でも固定プロンプトの数値ゲートを通過。逐次待ちの対照よりdecodeを改善した。行分割自体の高速化・最大モデル容量の増加は未確認。**

後続の小さな検査で非同期読み出しの失敗を再現し、既存の同期モードを選ぶ対策を追加した。[転送の再現例と現在の対応範囲](readback.md)を先に参照。以下のゼロ出力・画面停止は、その対策前の履歴として残している。

2026-11-03、Pixel 11 Proの標準Debian VM、PowerVR、llama.cpp b11960を使用。ソースはv0.8.0、コミット `8642541b255bfb5dc39cd351a178be09013a55d6`。既存のcore-fp16ランタイムは保存し、別の作業ディレクトリで調べた。[実機ログと出力値](../examples/inference/output-row-study/)

## CPUに残る処理を特定する

    ./pixel-lab run inference --runtime /absolute/checked-runtime --schedule-report

doctorの整数照合に続き、タイミング測定とは別のtg1プローブで `GGML_SCHED_DEBUG=2` を使用する。`scheduler.log`、`scheduler-bench.json`、`schedule.json` にコマンド・生ログのハッシュ・最後のグラフの配置を保存する。プローブの失敗や空の配置を成功扱いしない。通常の速度測定にはこのデバッグ設定を渡さない。

既存0.5Bファイルの `output.weight` はQ8_0、896×151936、約137.94 MiB。128 MiBのstorage buffer rangeを超えるためVulkan用のバッファを選べず、CPU_REPACKへ退避し、最終 `result_output` のMUL_MATがCPUに残っていた。入力埋め込みのGET_ROWSもCPU。ログの「25/25 layers offloaded」だけでは、全演算がGPUにあるとは判断できない。

この配置の特定は、最終行列積が性能上の最大要因であることの証明ではない。

## 分割候補の範囲

`GGML_VK_SPLIT_OUTPUT=1` のときだけ、Qwen2の大きな出力行列を行方向のviewに分け、複数のMUL_MATとCONCATを作る。量子化済みの重みは変更しない。1片は64 MiB以下・65536行以下、境界は256行単位。モデルの出力重みが実際にVulkanバッファにある場合だけ使い、LoRAの併用は明示的に拒否する。

メタデータだけのviewに限って親テンソルのstorage rangeチェックを緩和し、実際の計算の範囲チェックは残す。アップロードは16 MiBずつ同期して行う。ホスト読み出しのbarrierとstaging領域のinvalidateも調査パッチに含めた。これらでPowerVRドライバの不具合が証明されたわけではない。

分割後のスケジュールでは3個の出力行列積とCONCATをVulkanに配置できた。ただし配置だけでは計算結果の正常性は保証できない。

## 対策前の検査で分かったこと

| 検査 | 観測 | 扱い |
| --- | --- | --- |
| 約138 MiB、Q8_0、896×151937、N=1/16の単体グラフ | 重みを別バッファに置いたbasis入力で2回合格。22箇所の転送照合と分割境界を検査 | 転送・索引・CONCATの限定的な証拠 |
| 同一バッファに重みと計算領域を置いた先行検査 | ゼロ出力、NMSE=1。barrier追加だけでは改善せず | 失敗を保存。汎用の大行列対応とは扱わない |
| 実モデルの固定33トークン入力 | 分割時、151936個のlogitすべてがゼロ。関連ライブラリとツールを同じソースでそろえても再現 | 採用不可 |
| テンソルを途中回収するcallback診断 | 完了前に端末GUIが黒く停止。Androidのwatchdog記録を観測 | 診断未完了。原因の断定をしない |
| 復旧 | 通常のADB再起動後、協力者がロック画面の表示と解除を確認 | この時点でGPU実験を停止。後続は小さな転送検査から再開 |

basis入力は独立に復号したQ8_0の値を参照する。密な入力での総和の正しさや、実モデル全体の正しさまでは検証しない。CPU対basisの誤差も記録するが、その値はこの単体検査の合否条件には含めない。

旧診断ツールのJSONにはゼロ出力でも `status: pass` が残る。この旧判定は処理終了と有限値だけを確認していたため、研究結果としては失敗。元のJSONを上書きせず、新しい検査で全ゼロを拒否する。

## 再開用のビルドと検査

端末の通常利用を優先し、実験用ランタイムは日常用のものと別に保存する。次のコマンドは行分割と完了待ちのパッチを組み込む。

    sh guest/build-inference-runtime.sh /absolute/fresh/llama.cpp-0.8.0 /absolute/stock-runtime /absolute/experimental-runtime --output-row-split

分割モードではCPU・Vulkan・llama・common・ベンチマークと生成ツールを同じソースからO1・1並列でビルドする。既存のCPU派生プラグインを混ぜず、別のステージング先で数値検査を通した後に移動する。ビルド記録は「実モデル未検証」を明記する。モデルを指定していないビルドの完了は、実モデルの合格を意味しない。

    ./pixel-lab run inference --runtime /absolute/experimental-runtime --model /absolute/model.gguf --gpu-layers 99 --split-output

このモードは同期読み出しとキュー完了待ちを選び、最初に最大1 MiBの転送27ケースを照合する。ライブラリと検査ツールのハッシュ、既存56ケース、大行列のbasis検査に加え、同じモデル・同じランタイムでCPU参照とGPU候補の全logitを保存・比較してから速度を測る。全ゼロ・非有限値・件数不一致を拒否し、暫定条件としてNMSE≤0.005とtop1一致を要求する。Flash Attention off・repack 1に限定する。出力行列のみをGPUへ置く要求は1。

`model-reference.bin`、`model-candidate.bin` はlittle-endian float32。各JSON・ログ・`model-check.json` に条件、出力ハッシュ、比較値を残す。この条件は1プロンプトの限定的な数値検査であり、回答品質・全プロンプト・局所的な誤差の不存在を保証しない。復旧後は小さな転送検査から再開し、完了待ちの修正後に全GPU要求でもdoctorから統合経路を完走した。

先行して用意した[stagingへ迂回させる追加案](../experiments/output-row-held/)はビルド対象には含めていない。通常ホスト配列の非同期読み出しでも古い値が出たため、pinされた出力先だけを問題の原因とは断定できない。同期getだけでは後の行列積検査も失敗した。現在は[キュー完了待ちを加えた経路](readback.md)で、全GPU要求の固定入力を照合している。今後も実モデルの照合を先に通してから速度を比べる。
