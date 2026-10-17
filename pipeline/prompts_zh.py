"""Chinese text for the multi-mode check (user request, 2026-10-17): the template and one question per rule.
The answer lines stay "<k>. YES" / "<k>. NO" and the observation stays in Russian."""

MULTI_TEMPLATE_ZH = """这是银行收银台监控摄像头拍摄的一段短视频。
请用 YES 或 NO 回答以下 {n} 个关于该视频的问题。只有在视频中清楚可见时才回答 YES。
{questions}

回答要求：
- 只根据视频中可见的内容作答。不要猜测金额、文件、签名、账户记账或其他看不到的内容。
- 不要说出或试图识别任何人的身份，也不要描述面部或外貌。只用以下角色称呼人物：«кассир»、«клиент» 或 «инкассатор»。

请严格按以下格式回复，共 {lines} 行：每个问题一行，顺序相同，答案只写英文 YES 或 NO；最后一行是观察：
{answers}
OBSERVATION: <用俄语（русский язык）写 1-3 句话，只描述可见的动作，包括在场人员（只写角色）以及现金所在位置>"""

VERIFY_PROMPTS_ZH: dict[str, str] = {
    "cash_unattended": "收银员离开工作位时，是否把现金暴露地留在柜台上？",
    "cashier_leaves_during_service": "在仍在为客户办理业务时，收银员是否离开了工作位？",
    "previous_cash_on_desk": "在为客户办理业务时，收银员是否把之前交易的其他现金敞放在柜台上，而不是放在抽屉或保险柜里？",  # noqa: E501
    "cash_storage_left_open": "在无人看管的情况下，现金抽屉、保险柜或钱箱是否敞开，或钥匙是否留在锁上？",
    "cash_into_personal_belongings": "是否有人把纸币放进口袋、衣服或个人包里，而不是放进现金抽屉或保险柜？",
    "personal_items_at_workplace": "收银员是否在收银工作位使用手机，或在那里放置手提包等个人物品？",
    "deposit_not_counted": "收银员是否收下客户的现金后，未经手工或点钞机清点就直接收起来？",
    "deposit_counted_without_client": "客户的现金是否在客户离开之后、或在客户看不到的地方才清点？",
    "withdrawal_not_counted": "收银员是否未当着客户的面清点，就把散装或未封装的现金交给客户？",
    "recount_without_controller": "在没有接待客户时，收银员是否在没有第二名员工监督的情况下独自清点、整理或打包现金？",
    "unauthorized_person_in_cash_area": "是否有非收银员的人进入收银员的工作区，或把手伸进现金区触碰现金？",
    "collection_cash_not_in_sealed_bag": "收银员与押运员之间交接的现金是否是散装的，或装在敞口、未封口的袋子里？",
    "collection_bag_not_inspected": "收银员或押运员是否在没有明显检查封条和完好性的情况下接收押运袋？",
}
