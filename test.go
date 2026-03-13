package main
import (
	"fmt"
	"strings"
)

func reverseWords(s string) string {
	words := strings.Fields(s)
	left, right := 0, len(words)-1
	for left < right {
		words[left], words[right] = words[right], words[left]
		left++
		right--
	}
	return strings.Join(words, " ")
}
func main() {
	s := "  hello world  "
	result := reverseWords(s)
	fmt.Printf("Reversed string: '%s'\n", result)
}