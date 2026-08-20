"""OpenAI 流式 tool_calls 增量聚合测试。"""

from infinity_agent.clients.open_ai.aggregation import aggregate_tool_call_deltas
from infinity_agent.clients.open_ai.response_models import ToolCallDelta, ToolCallFunction


def _delta(
    index: int,
    *,
    id: str | None = None,
    name: str | None = None,
    arguments: str = '',
) -> ToolCallDelta:
    return ToolCallDelta(
        index=index,
        id=id,
        function=ToolCallFunction(name=name, arguments=arguments),
    )


class TestAggregateToolCallDeltas:
    """工具调用增量聚合"""

    def test_single_delta(self) -> None:
        deltas = [
            _delta(0, id='call_1', name='get_weather', arguments='{"city":"北京"}')
        ]
        result = aggregate_tool_call_deltas(deltas)
        assert len(result) == 1
        assert result[0].id == 'call_1'
        assert result[0].function.name == 'get_weather'
        assert result[0].function.arguments == {'city': '北京'}

    def test_arguments_concatenated_across_chunks(self) -> None:
        deltas = [
            _delta(0, id='call_1', name='get_weather', arguments='{"cit'),
            _delta(0, arguments='y":"上海"}'),
        ]
        result = aggregate_tool_call_deltas(deltas)
        assert len(result) == 1
        assert result[0].function.arguments == {'city': '上海'}

    def test_multiple_indexes_preserve_order(self) -> None:
        deltas = [
            _delta(1, id='call_2', name='b', arguments='{}'),
            _delta(0, id='call_1', name='a', arguments='{}'),
        ]
        result = aggregate_tool_call_deltas(deltas)
        assert [tc.id for tc in result] == ['call_1', 'call_2']

    def test_name_in_later_chunk_wins(self) -> None:
        """name 字段采用「后者覆盖前者」语义（首个 chunk 常为空，后续补齐）"""
        deltas = [
            _delta(0, id='call_1', arguments='{"city":'),
            _delta(0, name='get_weather', arguments='"北京"}'),
        ]
        result = aggregate_tool_call_deltas(deltas)
        assert result[0].function.name == 'get_weather'
        assert result[0].function.arguments == {'city': '北京'}

    def test_name_replaced_by_later_chunk(self) -> None:
        """多次出现的 name 由最后一个非空值决定"""
        deltas = [
            _delta(0, id='call_1', name='old', arguments='{}'),
            _delta(0, name='new', arguments='{}'),
        ]
        result = aggregate_tool_call_deltas(deltas)
        assert result[0].function.name == 'new'

    def test_missing_id_skipped(self) -> None:
        deltas = [_delta(0, name='f', arguments='{}')]
        assert aggregate_tool_call_deltas(deltas) == []

    def test_missing_name_skipped(self) -> None:
        deltas = [_delta(0, id='call_1', arguments='{}')]
        assert aggregate_tool_call_deltas(deltas) == []

    def test_empty_list(self) -> None:
        assert aggregate_tool_call_deltas([]) == []

    def test_invalid_json_arguments_becomes_empty_dict(self) -> None:
        deltas = [_delta(0, id='call_1', name='f', arguments='{not valid json')]
        result = aggregate_tool_call_deltas(deltas)
        assert result[0].function.arguments == {}

    def test_empty_arguments_becomes_empty_dict(self) -> None:
        deltas = [_delta(0, id='call_1', name='f')]
        result = aggregate_tool_call_deltas(deltas)
        assert result[0].function.arguments == {}

    def test_late_id_and_name_fill(self) -> None:
        """id/name 可能在后续 chunk 才出现"""
        deltas = [
            _delta(0, arguments='{"x":1}'),
            _delta(0, id='call_1', name='f'),
        ]
        result = aggregate_tool_call_deltas(deltas)
        assert len(result) == 1
        assert result[0].id == 'call_1'
        assert result[0].function.name == 'f'
        assert result[0].function.arguments == {'x': 1}
