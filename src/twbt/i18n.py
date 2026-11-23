"""English and 繁體中文 strings for the builder.

Two languages, one dictionary. The keys are English-ish mnemonics rather than
the English text itself, so a wording change in one language does not silently
orphan the other.

Why the terms are *not* translated: 漲停, 跌停, 平盤下不得放空, 證交稅, 手續費
and 借券費 appear in Chinese in both languages. They are the names of specific
Taiwanese market rules and fees, and an English reader who works on this market
knows them by those names — "price limit up" is a translation, not a term.
"""

from __future__ import annotations

LANGS = {"en": "English", "zh": "繁體中文"}
DEFAULT = "en"


def pick(lang: str) -> str:
    return lang if lang in LANGS else DEFAULT


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #

UI: dict[str, dict[str, str]] = {
    "subtitle":        {"en": "interactive cross-sectional backtest",
                        "zh": "互動式橫斷面回測"},
    "operators":       {"en": "Operators", "zh": "運算子"},
    "drag_hint":       {"en": "Drag onto the canvas, or double-click to append.",
                        "zh": "拖曳到畫布，或雙擊加到最後。"},
    "filter":          {"en": "filter…", "zh": "篩選…"},
    "two_matrix_note": {"en": "more operators take two matrices. They need the code tab — a chain of boxes has only one input.",
                        "zh": "個運算子需要兩個矩陣，只能在程式碼分頁使用——積木串接只有一個輸入。"},
    "blocks":          {"en": "Blocks", "zh": "積木"},
    "code":            {"en": "Code", "zh": "程式碼"},
    "run":             {"en": "Run backtest", "zh": "執行回測"},
    "clear":           {"en": "Clear", "zh": "清除"},
    "source":          {"en": "source", "zh": "資料來源"},
    "drop_here":       {"en": "Drag operators here", "zh": "把運算子拖到這裡"},
    "no_portfolio":    {"en": "No portfolio step yet — add LongShortBalance or GrossScale to make weights.",
                        "zh": "還沒有部位建構步驟——加入 LongShortBalance 或 GrossScale 才會變成權重。"},
    "steps":           {"en": "steps", "zh": "個步驟"},
    "what_you_write":  {"en": "What you can write", "zh": "可以寫什麼"},
    "sources_are":     {"en": "Sources", "zh": "資料來源"},
    "arith_note":      {"en": "Arithmetic + - * / and parentheses work on matrices, so Returns(px) / InstStdev(20, Returns(px)) is a volatility-scaled return.",
                        "zh": "加減乘除與括號都可用於矩陣，所以 Returns(px) / InstStdev(20, Returns(px)) 就是波動度調整後的報酬。"},
    "settings":        {"en": "Settings", "zh": "設定"},
    "exec_lag":        {"en": "execution lag", "zh": "執行延遲"},
    "lag_1":           {"en": "1 session (tradable)", "zh": "1 個交易日（可執行）"},
    "lag_n":           {"en": "sessions", "zh": "個交易日"},
    "lag_0":           {"en": "0 — same close (not tradable)",
                        "zh": "0——同一根收盤（實際做不到）"},
    "costs":           {"en": "Costs", "zh": "交易成本"},
    "brokerage":       {"en": "手續費 each side", "zh": "手續費（單邊）"},
    "tax":             {"en": "證交稅 (sell side only)", "zh": "證交稅（僅賣出）"},
    "spread":          {"en": "half spread each side", "zh": "半價差（單邊）"},
    "borrow":          {"en": "借券費 per year", "zh": "借券費（年化）"},
    "bps":             {"en": "bps", "zh": "bps"},
    "pct_yr":          {"en": "%/yr", "zh": "%/年"},
    "tradability":     {"en": "enforce 漲停/跌停 and 平盤下不得放空",
                        "zh": "套用漲停/跌停與平盤下不得放空"},
    "day_trade":       {"en": "當沖 tax rate", "zh": "當沖稅率"},
    "round_trip":      {"en": "round trip", "zh": "來回成本"},
    "reset_costs":     {"en": "reset", "zh": "回復預設"},
    "tax_note":        {"en": "證交稅 is charged on sells only, which is why a long round trip costs more than you would guess from a single rate.",
                        "zh": "證交稅只在賣出時課徵，所以做多來回的成本會比單一費率推估的高。"},
    "presets":         {"en": "Presets", "zh": "範例策略"},
    "verdict":         {"en": "Verdict", "zh": "結論"},
    "performance":     {"en": "Performance", "zh": "績效"},
    "is_it_real":      {"en": "Robustness checks", "zh": "穩健性檢定"},
    "what_it_costs":   {"en": "What it costs", "zh": "成本"},
    "tradability_h":   {"en": "Tradability", "zh": "可交易性"},
    "statistics":      {"en": "Statistics", "zh": "統計"},
    "running":         {"en": "Running…", "zh": "執行中…"},
    "done_in":         {"en": "Done in", "zh": "完成，耗時"},
    "add_operator":    {"en": "Add at least one operator.", "zh": "至少加入一個運算子。"},
    "write_expr":      {"en": "Write an expression first.", "zh": "請先輸入運算式。"},
    "valid_blocks":    {"en": "Valid — also editable as blocks.",
                        "zh": "語法正確——也可以用積木編輯。"},
    "valid_tree":      {"en": "Valid. This expression branches, so it has no block form.",
                        "zh": "語法正確。這個運算式有分支，無法用積木表示。"},
    "panel_unavailable": {"en": "Panel unavailable", "zh": "資料面板無法載入"},
    "fetch_first":     {"en": "Fetch the equity cross-section first:",
                        "zh": "請先抓取股票橫斷面資料："},
    "year":            {"en": "year", "zh": "年份"},
    "gross":           {"en": "gross", "zh": "毛"},
    "net":             {"en": "net", "zh": "淨"},
    "sharpe":          {"en": "Sharpe", "zh": "夏普值"},
    "hit":             {"en": "hit", "zh": "勝率"},
}

# chart titles
CHART: dict[str, dict[str, str]] = {
    "cum_pnl":      {"en": "Cumulative P&L", "zh": "累積損益"},
    "long_short":   {"en": "Long / short", "zh": "多空分解"},
    "by_year":      {"en": "Cumulative P&L by year", "zh": "各年度累積損益"},
    "avg_path":     {"en": "Average path by year", "zh": "年度平均走勢"},
    "lag_decay":    {"en": "Cumulative P&L by execution lag", "zh": "不同執行延遲的累積損益"},
    "buckets":      {"en": "Cumulative P&L by signal quintile", "zh": "訊號五分位累積損益"},
    "drawdown":     {"en": "Rolling drawdown", "zh": "滾動回撤"},
    "boards":       {"en": "Cumulative P&L by board", "zh": "上市／上櫃累積損益"},
    "cost":         {"en": "Return and cost", "zh": "報酬與成本"},
    "turnover":     {"en": "Turnover (20d average)", "zh": "週轉率（20 日平均）"},
    "counts":       {"en": "Names held", "zh": "持有檔數"},
    "limit_drag":   {"en": "Cost of 漲跌停 / 平盤下不得放空", "zh": "漲跌停與平盤下不得放空的代價"},
    "blocked":      {"en": "Blocked names per day", "zh": "每日無法成交檔數"},
    "untraded":     {"en": "Book weight that could not be traded (20d avg)",
                     "zh": "無法成交的權重（20 日平均）"},
    "y_cum_return": {"en": "cumulative return", "zh": "累積報酬"},
    "y_names":      {"en": "names", "zh": "檔數"},
    "y_fraction":   {"en": "fraction of gross", "zh": "佔總曝險比例"},
    "x_session":    {"en": "session of year", "zh": "年內第幾個交易日"},
    "l_total":      {"en": "total", "zh": "合計"},
    "l_long":       {"en": "long", "zh": "多方"},
    "l_short":      {"en": "short", "zh": "空方"},
    "l_gross":      {"en": "gross", "zh": "毛"},
    "l_net":        {"en": "net", "zh": "淨"},
    "l_cost":       {"en": "cost", "zh": "成本"},
    "l_intended":   {"en": "intended", "zh": "原本想要"},
    "l_tradable":   {"en": "tradable", "zh": "實際可做"},
    "l_turnover":   {"en": "turnover", "zh": "週轉率"},
    "l_drawdown":   {"en": "drawdown", "zh": "回撤"},
    "l_untraded":   {"en": "untraded", "zh": "未成交"},
    "l_lag":        {"en": "lag", "zh": "延遲"},
}

# figure captions
CAPTION: dict[str, dict[str, str]] = {
    "equity":     {"en": "Gross vs net. The gap is the cost.",
                   "zh": "毛報酬與淨報酬，兩條線的差距就是成本。"},
    "decomp":     {"en": "One side carrying everything is usually a direction bet.",
                   "zh": "績效若全來自單邊，通常只是押注市場方向。"},
    "by_year":    {"en": "Each year with its own Sharpe; dashed line is the average.",
                   "zh": "每年一條線並標註該年夏普值，虛線為平均走勢。"},
    "avg_path":   {"en": "What a typical year looked like, by side.",
                   "zh": "典型年度的走勢樣貌，並分多空兩邊檢視。"},
    "lag_decay":  {"en": "Gone by lag 2 means it needed the close it was computed from.",
                   "zh": "若延遲 2 日即消失，代表必須在計算當下的收盤成交，實務上做不到。"},
    "buckets":    {"en": "Long/short assumes monotonicity. Often it is one tail.",
                   "zh": "多空架構假設單調關係，但效果往往僅來自單一尾端。"},
    "drawdown":   {"en": "Depth and, more importantly, how long it stays down.",
                   "zh": "除了回撤深度，更要看在水面下停留多久。"},
    "boards":     {"en": "上市 vs 上櫃, with each board's share of gross.",
                   "zh": "上市與上櫃比較，括號為各自佔總曝險比例。"},
    "cost":       {"en": "Cumulative cost against the edge it is eating.",
                   "zh": "累積成本與它吃掉的報酬。"},
    "turnover":   {"en": "Turnover is what the 證交稅 is charged on.",
                   "zh": "週轉率就是證交稅的課徵基礎。"},
    "counts":     {"en": "Names held per side.", "zh": "多空各持有幾檔。"},
    "limit_drag": {"en": "The book you intended vs the one you could hold.",
                   "zh": "原先意圖建立的部位，與實際可持有的部位。"},
    "blocked":    {"en": "Names you wanted to trade and could not, per day.",
                   "zh": "每日意圖交易但無法成交的檔數。"},
    "untraded":   {"en": "Exposure the locks kept you out of.",
                   "zh": "因漲跌停與放空限制而未能建立的曝險。"},
}

# statistics table labels
STAT: dict[str, dict[str, str]] = {
    "net Sharpe":         {"en": "net Sharpe", "zh": "淨夏普值"},
    "gross Sharpe":       {"en": "gross Sharpe", "zh": "毛夏普值"},
    "t-stat":             {"en": "t-stat", "zh": "t 統計量"},
    "annualised return":  {"en": "annualised return", "zh": "年化報酬"},
    "max drawdown":       {"en": "max drawdown", "zh": "最大回撤"},
    "hit rate":           {"en": "hit rate", "zh": "勝率"},
    "turnover / day":     {"en": "turnover / day", "zh": "每日週轉率"},
    "cost / gross edge":  {"en": "cost / gross edge", "zh": "成本佔毛報酬"},
    "names held":         {"en": "names held", "zh": "平均持有檔數"},
    "sessions":           {"en": "sessions", "zh": "交易日數"},
}

# verdict sentences, as templates filled by xsim.verdict()
VERDICT: dict[str, dict[str, str]] = {
    "same_close":   {"en": "SAME-CLOSE EXECUTION: this is not tradable. A signal from today's close cannot be executed at today's close.",
                     "zh": "同收盤執行：實務上無法成交。以今日收盤計算的訊號，不可能在同一根收盤成交。"},
    "no_edge":      {"en": "FAILS: no edge even before costs.",
                     "zh": "未通過：尚未扣除成本即無超額報酬。"},
    "cost_kills":   {"en": "FAILS on cost: gross is positive but costs take {share:.0%} of it. Turnover is {turn:.1%} of gross per day.",
                     "zh": "未通過（成本）：毛報酬為正，但成本侵蝕其中 {share:.0%}。每日週轉率為總曝險的 {turn:.1%}。"},
    "net_positive": {"en": "Net positive: Sharpe {sharpe:.2f}, t={t:+.2f}, costs eat {share:.0%} of gross.",
                     "zh": "淨報酬為正：夏普值 {sharpe:.2f}，t={t:+.2f}，成本侵蝕毛報酬的 {share:.0%}。"},
    "years":        {"en": "{pos}/{n} calendar years net positive.",
                     "zh": "{n} 個年度中有 {pos} 年淨報酬為正。"},
    "one_year":     {"en": "A result carried by one year is one observation, not several.",
                     "zh": "績效集中於單一年度，實質上僅為一個觀察值，而非多個。"},
    "one_side":     {"en": "Almost all P&L comes from the {side} side (long {lng:+.3f} vs short {sht:+.3f}). Check it is not a market-direction bet.",
                     "zh": "損益幾乎全來自{side}方（多方 {lng:+.3f}、空方 {sht:+.3f}）。請確認並非單純押注市場方向。"},
    "tradability":  {"en": "Tradability rules (漲停/跌停, 平盤下不得放空) changed {moved:.1%} of intended book weight.",
                     "zh": "可交易性規則（漲停/跌停、平盤下不得放空）更動了 {moved:.1%} 的原始部位權重。"},
    "side_long":    {"en": "long", "zh": "多"},
    "side_short":   {"en": "short", "zh": "空"},
    "empty":        {"en": "Empty pipeline: the raw source is being used as weights.",
                     "zh": "流程為空：直接以原始資料作為權重。"},
    "no_portfolio": {"en": "No portfolio step (LongShortBalance / GrossScale / TopN). The engine will normalise anyway, but the book you get is not the one you specified.",
                     "zh": "沒有部位建構步驟（LongShortBalance / GrossScale / TopN）。引擎仍會自行標準化，但得到的部位不是你指定的那個。"},
    "rank_demean":  {"en": "Demean after Rank re-centres an already-uniform distribution, which mostly just shifts it by 0.5.",
                     "zh": "在 Rank 之後再 Demean，等於把已經均勻的分布再置中，實際上只是平移 0.5。"},
    "double_gross": {"en": "GrossScale more than once: all but the last are discarded.",
                     "zh": "重複使用 GrossScale：除了最後一次以外都會被覆蓋。"},
}

# operator hints. Only the Chinese side lives here; English is on the Op itself,
# next to the function, where it is most likely to be kept true.
OP_HINT_ZH: dict[str, str] = {
    "Lag": "把訊號往後推 n 天。用來讓訊號可執行，或測試延遲後優勢是否還在。",
    "Lead": "往前看 n 天。只能用來建立「要預測的目標」，絕不可放進訊號裡。",
    "InstMean": "每檔股票自己最近 n 天的平均，用來平滑雜訊。",
    "InstStdev": "每檔股票自己最近 n 天的波動度，風險調整常用的分母。",
    "InstSkew": "每檔股票最近 n 天的偏態，像樂透的股票分數高。",
    "InstSum": "每檔股票最近 n 天的加總。InstSum(5, Returns(px)) 就是上週報酬。",
    "InstCount": "最近 n 天實際有資料的天數，用來剔除資料太少的股票。",
    "InstRank": "今天在「這檔股票自己」最近 n 天中的位置，0 到 1。問的是「對這檔來說異常嗎？」",
    "InstZscore": "今天距離自己平均幾個標準差，可跨股票比較。",
    "InstBeta": "每檔股票對大盤的滾動 beta。基準傳入矩陣即可，會取其橫斷面平均當大盤。",
    "ForwardFill": "把最後一個值往後補，最多 n 天。有上限，避免已下市的股票變成永久部位。",
    "Returns": "逐日報酬率，最常見的起點。",
    "LogReturns": "對數報酬，跨時間可直接相加。",
    "Rank": "把每一天的所有股票互相比較後排名，0 到 1。橫斷面最常用的運算子。",
    "Demean": "減掉當天所有股票的平均，去掉大盤走勢，只留下相對強弱。",
    "Zscore": "每天在橫斷面做標準化。類似 Rank 但保留幅度，極端值仍然突出。",
    "Winsorize": "把每天的橫斷面在指定分位數截斷，壓制極端值但不刪除。",
    "Neutralize": "每天把訊號對某因子做迴歸並取殘差。回答「扣掉顯而易見的解釋後還剩什麼？」",
    "DemeanByGroup": "減掉所屬群組的平均，而不是大盤平均。區分「這檔便宜」與「整個產業便宜」。",
    "GroupRank": "在所屬群組內排名，而不是跟全市場比。",
    "Abs": "去掉正負號，只看大小。",
    "Log": "自然對數，壓縮像成交值這種長尾資料。非正值會變成空值。",
    "Sqrt": "平方根，比 Log 溫和的壓縮。",
    "Sign": "只保留 -1、0、+1，丟掉幅度——這常常正是你要的。",
    "Neg": "翻轉正負號。把動能訊號變成反轉訊號。",
    "Pow": "取次方。大於 1 放大極端值，小於 1 則壓平。",
    "Clip": "截斷到固定數值範圍。和 Winsorize 不同，這裡是絕對值而非分位數。",
    "Finite": "把無限大換成空值。除法之後記得加這個。",
    "Div": "矩陣相除，分母為零時給空值而不是無限大。",
    "And": "兩個輸入都非零時為 1，用來組合兩個條件。",
    "Or": "任一輸入非零時為 1。",
    "Mask": "只在第二個矩陣非零的地方保留第一個矩陣，其餘留空。",
    "LongShortBalance": "每天調整讓多空兩邊相等，部位就不再是對市場方向的押注。",
    "GrossScale": "每天縮放讓部位總和固定，損益才反映觀點而非部位大小。",
    "TopN": "每天只留最好和最差的各 n 檔。部位更集中——平均報酬更高，波動也更大。",
}


def t(table: dict, key: str, lang: str) -> str:
    """Look up one string, falling back to English then to the key itself."""
    row = table.get(key)
    if not row:
        return key
    return row.get(pick(lang)) or row.get(DEFAULT) or key


def op_hint(name: str, english: str, lang: str) -> str:
    if pick(lang) == "zh":
        return OP_HINT_ZH.get(name, english)
    return english
