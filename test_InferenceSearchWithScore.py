import pytest
from inference import search_with_score
from unittest.mock import Mock, call


class Test_InferenceSearchWithScore:

    @pytest.mark.regression
    def test_search_with_score_returns_correct_output(self):
        # Arrange
        mock_db = Mock()
        mock_db.similarity_search_with_score.return_value = [('doc1', '0.8'), ('doc2', '0.9')]
        query = 'Python programming tips'
        expected_output = 'expected output'

        # Act
        actual_output = search_with_score(mock_db, query)

        # Assert
        assert actual_output == expected_output

    @pytest.mark.negative
    def test_search_with_score_no_match(self):
        # Arrange
        mock_db = Mock()
        mock_db.similarity_search_with_score.return_value = []
        query = 'Some non-matching string'

        # Act
        actual_output = search_with_score(mock_db, query)

        # Assert
        assert actual_output == 'No match found'

    @pytest.mark.valid
    def test_search_with_score_method_calls(self):
        # Arrange
        mock_db = Mock()
        mock_db.similarity_search_with_score.return_value = [('doc1', '0.8'), ('doc2', '0.9')]
        query = 'Python programming tips'

        # Act
        search_with_score(mock_db, query)

        # Assert
        expected_calls = [
            call.similarity_search_with_score(query),
            call.as_retriever(),
        ]
        mock_db.assert_has_calls(expected_calls, any_order=False)

    @pytest.mark.invalid
    def test_search_with_score_db_errors(self):
        # Arrange
        mock_db = Mock()
        mock_db.similarity_search_with_score.side_effect = Exception('DB retrieval error')
        query = 'Python programming tips'

        # Act
        with pytest.raises(Exception) as e:
            search_with_score(mock_db, query)

        # Assert
        assert str(e.value) == 'DB retrieval error'
