// Export X/Twitter bookmarks as JSON
//
// Usage:
//   1. Go to https://x.com/i/bookmarks in your browser
//   2. Open DevTools (Cmd+Option+J / Ctrl+Shift+J)
//   3. Paste this entire script into the Console and press Enter
//   4. Wait while it auto-scrolls through your bookmarks
//   5. A bookmarks.json file will download automatically when done

;(() => {
  const links = new Set()
  const tweetUrlRegex = /^https:\/\/x\.com\/[^/]+\/status\/\d+$/
  const scrollStep = 4000
  const scrollInterval = 1200
  let unchanged = 0
  let prevCount = 0

  function collectLinks() {
    document
      .querySelectorAll('article[data-testid="tweet"] a[href*="/status/"]')
      .forEach((a) => {
        const url = a.href.split('?')[0]
        if (tweetUrlRegex.test(url)) links.add(url)
      })
  }

  function download() {
    const data = JSON.stringify([...links], null, 2)
    const blob = new Blob([data], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'bookmarks.json'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    console.log(`Done! ${links.size} bookmark links saved to bookmarks.json`)
  }

  const interval = setInterval(() => {
    collectLinks()
    const current = links.size
    console.log(`Links captured so far: ${current}`)

    if (current === prevCount) {
      unchanged++
      if (unchanged >= 4) {
        clearInterval(interval)
        console.log('Scrolling complete, downloading...')
        download()
        return
      }
    } else {
      unchanged = 0
    }

    prevCount = current
    window.scrollBy(0, scrollStep)
  }, scrollInterval)

  console.log('Started capturing bookmarks. Sit tight...')
})()
