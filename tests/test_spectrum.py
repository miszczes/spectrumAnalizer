
import spectrum
def test_stereoToMono():
    #Setup
    input = [[0, 5]]
    output = [3]

    #excersise
    result = spectrum.StereoToMono(input)

    #verify
    assert result == output